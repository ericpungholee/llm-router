"""Reproduce the frozen hard-pilot report offline; never mutate its input CSV."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_dataset import read_results, saved_model_blocks, is_completed_result
from model_registry import enabled_models
from pilot_analysis import analyze_results, _metrics, model_key, model_label, response_cost

SNAPSHOT = 'reports/snapshots/hard_pilot_final.csv'
PROVENANCE = 'reports/snapshots/hard_pilot_final.provenance.json'
RESUME = ('python3 generate_dataset.py --hard-pilot --resume --max-spend-usd 2.00 '
          '--output data/results/hard_pilot_results.csv')


def disagreement(rows, prompts, models):
    matrix = {(r['prompt_id'], model_key(r)): r for r in rows}
    common = [p for p in prompts if all((p, m) in matrix and matrix[p, m]['status'] == 'success' for m in models)]
    correct = sum(all(matrix[p, m]['correct'] for m in models) for p in common)
    wrong = sum(not any(matrix[p, m]['correct'] for m in models) for p in common)
    disagree = len(common) - correct - wrong
    return dict(n=len(common), all_correct=correct, all_wrong=wrong, all_agree=correct+wrong,
                disagree=disagree, rate=disagree/len(common) if common else None)


def static_metrics(rows, common, models):
    result = {}
    for m in models:
        selected = [r for r in rows if r['prompt_id'] in common and model_key(r) == m]
        costs = [response_cost(r) for r in selected]
        total = sum(costs, Decimal(0)) if selected and all(c is not None for c in costs) else None
        accuracy = sum(r['correct'] for r in selected)/len(selected) if selected else None
        result[model_label(m)] = dict(n=len(selected), accuracy=accuracy,
            total_cost_usd=float(total) if total is not None else None,
            average_cost_usd=float(total/len(selected)) if total is not None else None,
            average_latency_ms=mean(r['latency_ms'] for r in selected) if selected else None,
            correct_per_dollar=float(Decimal(str(accuracy))*len(selected)/total) if total else None)
    return result


def classification(row):
    if row['error_type'] == 'spend_limit_error': return 'spend-limit stop'
    if row['status'] == 'skipped_model': return 'skipped historical row'
    if row['status'] == 'parsing_failure': return 'parsing failure'
    if row['status'] == 'grading_failure': return 'grading failure'
    if row['error_type'] in {'invalid_model_error', 'configuration_error', 'authentication_error'}:
        return 'permanent provider/model failure'
    if row.get('failure_retryable') or row['error_type'] == 'network_error':
        return 'transient provider failure'
    return 'other'


def build_report():
    provenance = json.loads((ROOT / PROVENANCE).read_text())
    path = ROOT / SNAPSHOT
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['final_artifact']['sha256']:
        raise ValueError('Frozen snapshot SHA-256 differs from provenance')
    rows = read_results(path)
    plan = json.loads((ROOT / 'benchmarks/hard_pilot.json').read_text())['prompts']
    membership = {p:b for b, ps in plan.items() for p in ps}
    models = [(m.inference_provider, m.api_model_identifier) for m in enabled_models()]
    base = analyze_results(rows, list(membership), models, membership)
    common = base['routing_comparison']['comparison_prompt_ids']
    matrix = {(r['prompt_id'],model_key(r)):r for r in rows}
    vectors = []
    for p,b in membership.items():
        values = {model_label(m):matrix[p,m]['correct'] if (p,m) in matrix and matrix[p,m]['status']=='success' else None for m in models}
        vectors.append(dict(prompt_id=p,benchmark=b,correctness=values,
            statuses={model_label(m):matrix[p,m]['status'] if (p,m) in matrix else 'unrecorded' for m in models},
            confirmed_disagreement={v for v in values.values() if v is not None} == {True,False}))
    pairwise = [{'models':[model_label(a),model_label(b)], 'subsets':{
        name:disagreement(rows, ps, [a,b]) for name,ps in [('all',list(membership))]+list(plan.items())}}
        for a,b in combinations(models,2)]
    incomplete = [dict(prompt_id=r['prompt_id'],model=model_label(model_key(r)),
        classification=classification(r),status=r['status'],error_type=r['error_type'],reason=r['error_message'],
        has_saved_raw_response=bool(r['raw_response'].strip()), recoverable_offline=False,
        requires_provider_call=not bool(r['raw_response'].strip())) for r in rows if r['status']!='success']
    # This frozen study has no residual saved-response failures. Do not guess if replaced.
    assert all(not r['has_saved_raw_response'] for r in incomplete)
    secondary_models = [m for m in models if m[0] in ('openai','anthropic')]
    secondary_rows = [r for r in rows if model_key(r) in secondary_models]
    secondary = analyze_results(secondary_rows, list(membership), secondary_models, membership)
    secondary_common = secondary['routing_comparison']['comparison_prompt_ids']
    secondary['static_baselines'] = static_metrics(secondary_rows,secondary_common,secondary_models)
    secondary['disagreement'] = disagreement(secondary_rows,list(membership),secondary_models)
    comparison = secondary['routing_comparison']
    # Charge an explicit static fallback on unsolved prompts; the original oracle covers solved prompts only.
    fallback_label = comparison['always_cheapest_model']['model']
    fallback_rows = [r for r in secondary_rows if r['prompt_id'] in comparison['oracle_unsolved_prompt_ids'] and model_label(model_key(r))==fallback_label]
    full_cost = Decimal(str(comparison['oracle_cheapest_correct_cost_usd'])) + sum((response_cost(r) for r in fallback_rows),Decimal(0))
    comparison['oracle_cost_with_cheapest_static_fallback_usd'] = float(full_cost)
    comparison['oracle_full_cost_relative_to_best_static'] = float(full_cost/Decimal(str(comparison['always_best_single_model']['total_response_cost_usd'])))
    comparison['oracle_accuracy_lift'] = comparison['oracle_accuracy']-comparison['always_best_single_model']['accuracy']
    static = static_metrics(rows,common,models)
    cost_static = [(v['correct_per_dollar'],k) for k,v in static.items() if v['correct_per_dollar'] is not None]
    return dict(provenance=provenance,existing_analysis=base,overall=_metrics(rows,75),
        by_benchmark={b:_metrics([r for r in rows if r['benchmark']==b],len(ps)*len(models)) for b,ps in plan.items()},
        vectors=vectors,five_model_disagreement=disagreement(rows,list(membership),models),
        by_benchmark_disagreement={b:disagreement(rows,ps,models) for b,ps in plan.items()},
        confirmed_disagreement_lower_bound={'count':sum(v['confirmed_disagreement'] for v in vectors),'denominator':len(vectors)},
        pairwise_disagreement=pairwise,static_baselines=static,
        best_static_accuracy_cost_model=max(cost_static)[1] if cost_static else None,
        five_model_oracle={'accuracy':base['routing_comparison']['oracle_accuracy'],'lift_over_best_single':None,
            'cheapest_correct_cost_usd':base['routing_comparison']['oracle_cheapest_correct_cost_usd'],
            'cost_relative_to_best_static':None,'reason':'No fully graded five-model prompts'},
        incomplete_pairs=incomplete,incomplete_classifications=dict(Counter(r['classification'] for r in incomplete)),
        resume={'preflight_command':RESUME,'authorized_later_resume_command':RESUME+' --confirm',
            'eligible_pairs':sum(not is_completed_result(r) for r in rows),
            'persistent_model_blocks':len(saved_model_blocks(rows,enabled_models(),None)),
            'maximum_attempts_for_eligible_pairs':3*len(incomplete),'executed':False},
        secondary_gpt_claude=secondary,
        decision={'ml':'defer','routing_could_help':'Five-model complementarity unmeasurable. GPT/Claude accuracy shows static Claude dominance; a cost oracle suggests limited potential savings only.',
                  'learned_generalization':'Not established. A 15-prompt pilot cannot establish generalization.',
                  'reason':'24/75 grades, no five-model common subset, zero GPT/Claude oracle accuracy lift.'})


def render(report):
    lines=[]
    def add(s=''): lines.append(s)
    def fmt(x):
        if x is None:return 'N/A'
        if isinstance(x,float):return f'{x:.7f}'.rstrip('0').rstrip('.')
        return str(x)
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |'); add('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:add('| '+' | '.join(fmt(v).replace('|','\\|').replace('\n',' ') for v in row)+' |')
        add()
    def money(x):return 'N/A' if x is None else f'${x:.7f}'
    def short(label):return {'openai':'GPT','anthropic':'Claude','xai':'Grok','deepseek':'DeepSeek','openrouter':'Qwen'}[label.split('/')[0]]
    p=report['provenance'];m=report['overall'];base=report['existing_analysis'];c=base['routing_comparison']
    add('# Final hard-pilot routing analysis');add()
    add('**Decision: defer ML training.** The run is no longer active, but its final artifact has only 24/75 successfully graded pairs and no fully graded five-model prompts. Stopped does not mean successfully completed.');add()
    add('**Final artifact and provenance**');add()
    add(f"Source final CSV: `{p['source_final_csv']}`. Primary analysis input: [`{SNAPSHOT}`](snapshots/hard_pilot_final.csv). Frozen at {p['frozen_at_utc']}, from working tree based on commit `{p['base_commit']}`. The input CSV is included in the repository; no `data/results/` files or credentials are needed for reproduction.");add()
    add(f"Final SHA-256: `{p['final_artifact']['sha256']}`. File size: {p['final_artifact']['size_bytes']:,} bytes; row count: {p['final_artifact']['row_count']}.");add()
    obs=p['stability']['observations']
    add(f"Stability evidence: process inspection found no Python or hard-pilot writer. Identical size, SHA-256, row count and modification time were observed at {obs[0]['observed_at']} and {obs[1]['observed_at']}. No original exit log/status was available, so successful exit is not asserted. Before regrade the source was {p['before_offline_regrade']['size_bytes']:,} bytes, SHA-256 `{p['before_offline_regrade']['sha256']}`.");add()
    add('The earlier `data/results/hard_pilot_analysis_snapshot.csv` is superseded. The final recovered bytes happen to have the same hash as that intermediate snapshot; this final snapshot was copied directly from the stopped source after recovery, not reconstructed from the earlier snapshot. No new successful responses were added between those states.');add()
    add('Secret inspection found no API keys, authorization headers, credential values, environment contents, or private-key material. Pattern scans, private exact comparisons against local credential values, and schema/text inspection found no credentials. No sanitization or field removal was necessary. Normal provider request IDs remain.');add()
    add('**One offline recovery and complete change audit**');add()
    add('Ran once against the stable source: `'+p['offline_regrade']['command']+'`. The existing standalone bold-option parser remains unchanged and narrow. Claude’s saved `**F**` on `mmlu_pro:12015` becomes parsed `F`, matching reference `F`. Raw responses and all provider telemetry, costs, latencies, tokens, timestamps, request IDs and generation settings were verified unchanged.');add()
    table(['Prompt / model','Field','Before','After'],[[r['prompt_id']+' / '+r['api_model_identifier'],field,repr(v['before']),repr(v['after'])] for r in p['offline_regrade']['changed_rows'] for field,v in r['fields'].items()])
    add('**Completeness**');add()
    keys=['expected_pairs','recorded_pairs','successfully_graded_pairs','provider_failures','parsing_failures','grading_failures','skipped_pairs','unrecorded_pairs','spend_limit_stops','attempted_provider_calls','successful_provider_responses','failed_provider_attempts','total_recorded_cost_usd']
    table(['Metric','Stable source before recovery','Final recovered snapshot'],[[key,p['before_offline_regrade']['metrics'][key],m[key]] for key in keys])
    add('Recorded spend includes failed-attempt reserves; it is not independently verified provider billing. Legacy empty-response rows with zero recorded cost do not prove zero actual billing. Attempts/responses include recorded or legacy-inferred history. Failures and skips never become incorrect answers.');add()
    headers=['Group','Expected','Recorded','Graded','Provider fail','Parse fail','Grade fail','Skipped','Missing','Spend stops','Attempts','Responses','Spend USD']
    groupkeys=['expected_pairs','recorded_pairs','successfully_graded_pairs','provider_failures','parsing_failures','grading_failures','skipped_pairs','unrecorded_pairs','spend_limit_stops','attempted_provider_calls','successful_provider_responses','total_recorded_cost_usd']
    table(headers,[[short(k)]+[v[x] for x in groupkeys] for k,v in base['by_model'].items()])
    table(headers,[[k]+[v[x] for x in groupkeys] for k,v in report['by_benchmark'].items()])
    table(headers,[[b+' / '+short(k)]+[v[x] for x in groupkeys] for b,vs in base['by_benchmark_and_model'].items() for k,v in vs.items()])
    add('**Five-model routing signal**');add()
    add('Correctness vectors use 1/0 only for successfully graded answers; F = provider failure and S = skipped.');add()
    model_labels=list(report['vectors'][0]['correctness'])
    table(['Prompt']+[short(x) for x in model_labels],[[v['prompt_id']]+[int(v['correctness'][x]) if v['correctness'][x] is not None else {'provider_error':'F','skipped_model':'S'}.get(v['statuses'][x],'?') for x in model_labels] for v in report['vectors']])
    d=report['five_model_disagreement']
    table(['Subset','Eligible prompts','All correct','All wrong','All agree','Disagree','Disagreement rate'],[[name]+[v[k] for k in ['n','all_correct','all_wrong','all_agree','disagree','rate']] for name,v in [('All five models',d)]+list(report['by_benchmark_disagreement'].items())])
    add('The zero counts above describe an empty eligible subset, not zero disagreement across the 15-prompt plan. Full-plan totals are unknown. Partial vectors prove disagreement on at least 5/15 prompts (33.3%): two MMLU-Pro and three MATH-500; code has no grades.');add()
    add('Pairwise rows use each pair’s own jointly graded overlap. Denominators differ; N/A is not agreement.');add()
    table(['Pair','All','MMLU-Pro','MATH-500','LiveCodeBench'],[[' / '.join(short(x) for x in pair['models'])]+[f"{v['disagree']}/{v['n']} ({v['rate']:.0%})" if v['n'] else 'N/A (n=0)' for v in [pair['subsets'][b] for b in ['all','mmlu_pro','math_500','livecodebench']]] for pair in report['pairwise_disagreement']])
    add('**Static policies and oracle on the five-model common subset**');add()
    table(['Policy','n','Accuracy','Total cost','Average cost','Average latency ms'],[[short(k),v['n'],v['accuracy'],money(v['total_cost_usd']),money(v['average_cost_usd']),v['average_latency_ms']] for k,v in report['static_baselines'].items()])
    add('Best single model, cheapest model, best static accuracy/cost model, five-model oracle accuracy, oracle lift, cheapest-correct cost and oracle cost relative to best static are all N/A: the common subset is empty. No reduced comparison substitutes for these missing five-model results.');add()
    add('**Secondary diagnostic: GPT/Claude only, ten shared graded prompts**');add()
    s=report['secondary_gpt_claude'];sc=s['routing_comparison']
    table(['Model','n','Accuracy','Total cost','Average cost','Average latency ms'],[[short(k),v['n'],v['accuracy'],money(v['total_cost_usd']),money(v['average_cost_usd']),v['average_latency_ms']] for k,v in s['static_baselines'].items()])
    add('These ten prompts comprise five MMLU-Pro and five MATH-500; neither model has a code grade. They agree on six (four both correct, two both wrong) and disagree on four (40%). MMLU-Pro disagreement is 1/5; MATH-500 is 3/5. Every disagreement favors Claude. Claude is more accurate, cheaper in observed mean cost, and faster on this subset; it also has the best static accuracy/cost ratio. That is static dominance, not complementary accuracy strengths.');add()
    add(f"The reduced oracle is {sc['oracle_accuracy']:.0%}, with {sc['oracle_accuracy_lift']:.0%} accuracy lift over Claude. Cheapest-correct cost is {money(sc['oracle_cheapest_correct_cost_usd'])} for eight solved prompts only. The two unsolved prompts are `mmlu_pro:5022` and `mmlu_pro:6502`. Charging a Claude fallback on those gives {money(sc['oracle_cost_with_cheapest_static_fallback_usd'])} across all ten, {sc['oracle_full_cost_relative_to_best_static']:.1%} of Claude’s total cost (15.3% savings at equal accuracy). This uses ex-post correctness and realized costs; it does not demonstrate a prompt-visible policy or generalization.");add()
    add('**Why the matrix is incomplete and what resume would do**');add()
    add('All 51 incomplete pairs are listed below. None has a saved response, so none is recoverable offline; each would need another provider call. Five empty-text provider failures are classified as “other”, not permanent model failures or incorrect answers: the CSV does not establish a permanent cause. One network timeout is transient. All 45 historical skips followed an earlier provider/model failure. Parsing failures, grading failures, permanent failures and spend-limit stops are zero after recovery.');add()
    table(['Prompt','Model','Classification','Saved reason / error'],[[r['prompt_id'],short(r['model']),r['classification'],r['error_type']+': '+r['reason']] for r in report['incomplete_pairs']])
    add('Current safety logic identifies 51 eligible pending pairs and no persistent invalid-model/configuration blocks. The 24 paid responses remain terminal, including incorrect responses. Qwen’s non-retryable empty-response errors are not retried within that request, but their pairs are eligible on resume. Up to 153 attempts could be needed at two retries per pair. A $2 total cap includes the existing $0.3449374, leaving $1.6550626; guards may stop early and completion is not guaranteed.');add()
    add('Preflight only (no provider calls; not executed here):');add();add('```bash\n'+RESUME+'\n```');add()
    add('After explicit later authorization to spend, the exact resume command is:');add();add('```bash\n'+RESUME+' --confirm\n```');add()
    add('From a fresh clone, first restore a working resume CSV without overwriting any existing live results: `mkdir -p data/results` then `cp -n reports/snapshots/hard_pilot_final.csv data/results/hard_pilot_results.csv`. Never pass the frozen report snapshot as the live output.');add()
    add('**Decision and reproducibility**');add()
    add('Evidence that routing could help: partial disagreement exists and the reduced cost oracle leaves modest theoretical savings, but five-model complementary strengths cannot be assessed; the available accuracy evidence supports choosing Claude statically. Evidence that a learned router could generalize: none. Fifteen prompts cannot establish generalization, and no held-out learned policy was evaluated. Defer ML; resolve evaluation coverage before revisiting the decision. The existing analyzer’s broad `evidence_of_routing_signal` flag also counts static cheaper-model wins and is not used as the ML decision gate.');add()
    add('Test verification: '+json.dumps(p['verification'],sort_keys=True)+'.');add()
    add('From the repository root of a fresh clone, using Python 3.9+ (standard library is sufficient for these commands):');add()
    add('```bash\npython3 pilot_analysis.py reports/snapshots/hard_pilot_final.csv\npython3 reports/reproduce_hard_pilot.py\npython3 -m unittest discover -s tests -v\n```');add()
    add('The report command verifies the snapshot SHA and regenerates all three analysis files deterministically. It reads the tracked snapshot, provenance and fixed benchmark plan; it never reads the moving source, loads credentials, calls providers, or rewrites the snapshot. The provenance audit reconstructs the before-regrade CSV for verification. Keep this snapshot immutable; future paid outcomes belong to a separately versioned artifact.');add()
    return '\n'.join(lines).rstrip()+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    args=parser.parse_args()
    report=build_report()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in [('hard_pilot_existing_analysis.json',report['existing_analysis']),('hard_pilot_routing_analysis.json',report)]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (args.output_dir/'hard_pilot_routing_analysis.md').write_text(render(report))


if __name__=='__main__':
    main()
