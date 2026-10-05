"""Independent output searches and explicit input/output relation contracts.

Searches see frozen predictions as ordinary one-node features. The bundled
model replaces those features with opaque, validated ADF calls, preserving
local constraints without expanding an upstream tree into a downstream tree.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def solved_output_reached(afpo, args, islands, output_data):
    """Near-zero training and holdout loss, for the same frozen predictor."""
    if (not getattr(args,'_output_component',False)
            or getattr(args,'output_auto_advance','on') != 'on'):
        return False
    tolerance=getattr(args,'output_stop_at_loss',1e-12)
    Xv,Yv,cats,constraints,names=output_data
    checked=getattr(args,'_output_solved_checked',None)
    if checked is None:
        checked=args._output_solved_checked={}
    for island in islands:
        model=island.best_models.model
        if model is None or not model.feasible or not afpo.aggregate_loss(model)<=tolerance:
            continue
        key=afpo.selection_identity(model)
        if key not in checked:
            checked[key]=(Xv is None or afpo.frozen_metrics(model,Xv,Yv,cats,constraints,names)['loss']<=tolerance)
        if checked[key]:
            args._output_solved_model=model.clone()
            source='training and validation' if Xv is not None else 'training (no validation set)'
            print(f'Output {", ".join(names)} solved: {source} loss <= {tolerance:g}; advancing.')
            return True
    return False


def solved_output_choice(afpo, args, evaluation):
    """Select only genuinely solved candidates, without the MDL near-tie floor."""
    if getattr(args,'_output_solved_model',None) is None:
        return None
    tolerance=getattr(args,'output_stop_at_loss',1e-12)
    entries=[entry for entry in evaluation[1] if entry[0].feasible
             and afpo.aggregate_loss(entry[0])<=tolerance and entry[2]['loss']<=tolerance]
    return min(entries,key=lambda entry:(afpo.model_complexity(entry[0]),entry[2]['loss'])) if entries else None


def compile_input_groups(specifications, names):
    groups = []
    used = set()
    for specification in specifications:
        for text in specification.split(';'):
            members = [name.strip() for name in text.split(',') if name.strip()]
            if not members:
                raise ValueError('Empty input relation group')
            indices = set()
            for name in members:
                matches = [names.index(name)] if name in names else [i for i, feature in enumerate(names)
                           if feature.startswith(name + '=')]
                if not matches:
                    raise ValueError(f'Unknown input relation variable {name!r}')
                if indices.intersection(matches):
                    raise ValueError(f'Duplicate input relation variable {name!r}')
                indices.update(matches)
            if used.intersection(indices):
                raise ValueError('Input relation groups must be disjoint')
            used.update(indices)
            groups.append(frozenset(indices))
    if groups:
        groups.extend(frozenset((i,)) for i in range(len(names)) if i not in used)
    return tuple(groups)


def input_relation_violation(tree, groups, adfs=None, trusted_features=()):
    if not groups:
        return ''
    membership = {index: group for group, indices in enumerate(groups) for index in indices}

    adfs = adfs or {}
    def substitute(node, arguments):
        if node[0] == 'arg': return arguments[node[1]]
        if node[0] in ('x','c'): return node
        return (node[0], *(substitute(child,arguments) for child in node[1:]))

    def visit(node, visiting=frozenset()):
        if node[0] == 'x':
            group = membership.get(node[1])
            return {group: {node[1]}} if group is not None else {}, node[1] in trusted_features, ''
        if node[0] in ('c', 'arg'):
            return {}, True, ''
        if node[0].startswith('adf_'):
            item = adfs.get(node[0])
            if item and item.get('trusted_output'):
                return {}, True, ''  # validated staged outputs are opaque
            if item:
                if node[0] in visiting: return {}, False, 'input_relation:cyclic_adf'
                return visit(substitute(item['tree'],node[1:]),visiting | {node[0]})
            if len(node) == 1: return {}, True, ''
        children = [visit(child,visiting) for child in node[1:]]
        for _, _, reason in children:
            if reason:
                return {}, False, reason
        combined = {}
        for uses, _, _ in children:
            for group, indices in uses.items():
                combined.setdefault(group, set()).update(indices)
        if len(combined) > 1:
            for uses, formed, _ in children:
                if uses and (not formed or any(indices != set(groups[group]) for group, indices in uses.items())):
                    return combined, False, 'input_relation:cross_group_before_group_expression'
        return combined, True, ''

    uses, _, reason = visit(tree)
    if reason:
        return reason
    if any(indices != set(groups[group]) for group, indices in uses.items()):
        return 'input_relation:incomplete_group_expression'
    return ''


def output_dag(specifications, names):
    parents = {name: [] for name in names}
    for specification in specifications:
        for path in specification.split(';'):
            chain = [name.strip() for name in path.split('->')]
            if len(chain) < 2 or any(name not in parents for name in chain):
                raise ValueError(f'Invalid output relation {path!r}; use known outputs A->B')
            for earlier, later in zip(chain, chain[1:]):
                if earlier not in parents[later]:
                    parents[later].append(earlier)
    order = []
    while len(order) < len(names):
        ready = [name for name in names if name not in order and all(p in order for p in parents[name])]
        if not ready:
            raise ValueError('Output relations contain a cycle')
        order.extend(ready)
    return order, parents


def augment_predictions(afpo, X, parents, components):
    """Only prediction arrays enter derived features; this API accepts no targets."""
    if X is None:
        return None
    predictions = []
    for parent in parents:
        component = components[parent]
        inputs = augment_predictions(afpo, X, component['parents'], components)
        predictions.append(afpo.predict_model(afpo.Model(**component['model']), inputs))
    return np.column_stack([X, *predictions]) if predictions else X


def bundle_model(afpo, state):
    definitions = {}
    built = {}
    per_output = {}
    width = len(state['names'])
    for name in state['order']:
        component = state['components'][name]
        model = afpo.Model(**component['model'])
        replacements = {}
        offset = width
        for parent in component['parents']:
            for head in built[parent]:
                replacements[offset] = (head,)
                offset += 1
        rename = {old: f'adf_output_{state["out_names"].index(name)}_{old}' for old in model.adfs}

        def convert(tree):
            if tree[0] == 'x':
                return replacements.get(tree[1], tree)
            if tree[0] in ('c', 'arg'):
                return tree
            return (rename.get(tree[0], tree[0]), *(convert(child) for child in tree[1:]))

        for old, item in model.adfs.items():
            definitions[rename[old]] = {**item, 'tree': convert(item['tree'])}
        built[name] = []
        for head, (tree, scale) in enumerate(zip(model.trees, model.scales)):
            key = f'adf_output_{state["out_names"].index(name)}_head_{head}'
            definitions[key] = {'tree': convert(tree), 'arity': 0,
                                'output_scale': tuple(scale), 'trusted_output': True}
            built[name].append(key)
        per_output[name] = {'local_nodes': sum(afpo.node_size(t) for t in model.trees),
                            'local_head_nodes': [afpo.node_size(t) for t in model.trees],
                            'mdl_bits': afpo.model_complexity(model),
                            'history': list(model.history)}
    trees = [(key,) for name in state['out_names'] for key in built[name]]
    model = afpo.Model(trees, [(1., 0.)] * len(trees), adfs=definitions,
                       mdl_feature_count=width, mdl_operators=tuple(dict.fromkeys([*definitions,
                           *(op for c in state['components'].values() for op in c['model']['mdl_operators'])])))

    def expanded_size(tree, arguments=()):
        if tree[0] == 'arg':
            return expanded_size(arguments[tree[1]])
        if tree[0] in ('x', 'c'):
            return 1
        if tree[0] in definitions:
            item = definitions[tree[0]]
            def resolve(node):
                if node[0] == 'arg': return arguments[node[1]]
                if node[0] in ('x','c'): return node
                return (node[0], *(resolve(child) for child in node[1:]))
            size = expanded_size(item['tree'], tuple(resolve(child) for child in tree[1:]))
            # Fully inlined affine readout: a * f(x) + b.
            return size + (4 if 'output_scale' in item else 0)
        return 1 + sum(expanded_size(child, arguments) for child in tree[1:])

    for name in state['out_names']:
        per_output[name]['expanded_nodes'] = sum(expanded_size((key,)) for key in built[name])
    return model, {'per_output': per_output,
                   'total_expanded_nodes': sum(p['expanded_nodes'] for p in per_output.values())}


def save_bundle(afpo, path, state):
    destination = Path(path)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps({'format': 'afpo-independent-outputs-v1',
                                    'state': afpo._json_checkpoint_value(state)}, allow_nan=False))
    temporary.replace(destination)


def train_independent(afpo, args, setup, encoded, validation, fixture, ranges, choose_model, split, run_seed, external_validation):
    Xt, Yt, names, out_names, cats, maps = encoded
    order, parents = output_dag(args.output_relation, out_names)
    compile_input_groups(args.input_relation, names)
    seed = run_seed
    config = dict(vars(args))
    config.update(seed=seed)
    if config.get("test_csv"): config["test_csv"]=str(Path(config["test_csv"]).resolve())
    manifest = afpo.write_run_manifest(setup['path'], seed,
               {**config, 'independent_outputs': 'on', 'column_types': setup['types'],
                'max_nodes': setup['nodes'], 'max_depth': setup['depth'],
                'row_sample': setup['df'].attrs.get('afpo_row_sample')}, setup['df'],
               *split, external_validation=external_validation)
    state = {'args': config, 'setup': {key: value for key, value in setup.items() if key != 'df'},
             'Xt': Xt, 'Yt': Yt, 'Xv': validation[0], 'Yv': validation[1],
             'names': names, 'out_names': out_names, 'cats': cats, 'maps': maps,
             'fixture': fixture, 'ranges': ranges, 'order': order, 'parents': parents,
             'sample_seed': afpo.row_sample_seed(args), 'split': split, 'components': {}, 'manifest': str(manifest.resolve()), 'active_checkpoint': None}
    state['setup']['path'] = str(Path(setup['path']).resolve())
    if state['setup'].get('val_path') not in ('', '0', None):
        state['setup']['val_path'] = str(Path(state['setup']['val_path']).resolve())
    checkpoint = manifest.with_name('checkpoint_outputs.json').resolve()
    save_bundle(afpo, checkpoint, state)
    return run_bundle(afpo, args, state, checkpoint, choose_model)


def run_bundle(afpo, args, state, checkpoint, choose_model=None):
    from dataclasses import asdict
    save_bundle(afpo,checkpoint,state)
    for name in state['order']:
        if name in state['components']:
            continue  # upstream predictions remain frozen across resume
        index = state['out_names'].index(name)
        parents = state['parents'][name]
        derived_names = [f'predicted:{parent}:{head}' for parent in parents
                         for head in range(len(state['components'][parent]['model']['trees']))]
        if set(derived_names).intersection(state['names']):
            raise ValueError('A predicted feature name collides with an input column')
        names = [*state['names'], *derived_names]
        Xt = augment_predictions(afpo, state['Xt'], parents, state['components'])
        Xv = augment_predictions(afpo, state['Xv'], parents, state['components'])
        cat = [state['cats'][index]]
        encoded = (Xt, state['Yt'][:, index:index+1], names, [name], cat, state['maps'])
        valid = None if Xv is None else (Xv, state['Yv'][:, index:index+1], names, [name], cat, state['maps'])
        component_args = argparse.Namespace(**state['args'])
        component_args.max_generations = args.max_generations
        component_args.workers = args.workers
        component_args.independent_outputs = 'off'
        component_args.output_relation = []
        component_args.output_auto_advance=state['args'].get('output_auto_advance',getattr(args,'output_auto_advance','on'))
        component_args.output_stop_at_loss=state['args'].get('output_stop_at_loss',getattr(args,'output_stop_at_loss',1e-12))
        component_args.test_csv = None  # bundled test evaluation runs after every output is chosen
        component_args._output_component = True
        component_args._row_sample_seed = state["sample_seed"]
        # A reproducible stream per column, unaffected by any other search's duration.
        component_args.seed = (state['args']['seed'] + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'big')) % 2**32
        setup = dict(state['setup'])
        metadata = setup.get('metadata') or {}
        specs = metadata.get('outputs', metadata)
        setup['metadata'] = {'outputs': {name: specs.get(name, specs.get(str(index), {}))}}
        setup.update(_output_component=True, _prepared=(encoded, valid), _split=state['split'],
                     _trusted_feature_indices=list(range(len(state['names']),len(names))))
        setup['path'] = Path(setup['path'])
        setup['df'] = afpo.read_dataset(setup['path'], setup['delimiter'], component_args.max_rows,
                                        state['sample_seed'])
        print(f'Independent output search: {name}; predicted predecessors: {", ".join(parents) or "none"}')
        resume_path = state.get('resume_checkpoints',{}).get(name) or state.get('active_checkpoint')
        if resume_path:
            component_args.resume = resume_path
            _,_,_,_,saved=afpo.load_checkpoint(resume_path,component_args.allow_unsafe_pickle)
            if saved['out_names'] != [name] or saved['names'] != names:
                raise ValueError(f'Component checkpoint does not match output {name!r} and its predicted-feature contract')
            result = afpo.resume_main(component_args)
        else:
            # Register the per-search checkpoint as soon as it is written, so
            # a stop mid-search resumes its population rather than starting over.
            original_save = afpo.save_checkpoint
            def save_component(path, *positional, **keywords):
                original_save(path, *positional, **keywords)
                state['active_checkpoint'] = str(Path(path).resolve())
                save_bundle(afpo, checkpoint, state)
            afpo.save_checkpoint = save_component
            try:
                def choose_output(labels,choices,evaluation):
                    solved=solved_output_choice(afpo,component_args,evaluation)
                    if solved is not None:
                        selected=solved[0]
                        index=next((i for i,model in enumerate(choices) if model is selected),None)
                        if index is None:
                            labels.append('Solved output (near-zero training and validation loss)')
                            choices.append(selected); index=len(choices)-1
                        print(f'Automatically selected solved output {name}.')
                        return index
                    return (choose_model or afpo.choose_model_interactively)(labels,choices,evaluation)
                result = afpo.train_from_setup(component_args, setup, choose_output)
            finally:
                afpo.save_checkpoint = original_save
        state['components'][name] = {'model': asdict(result['model']), 'parents': parents,
                                      'checkpoint': result['checkpoint'], 'generation': result['generation']}
        state['active_checkpoint'] = None
        state.get('resume_checkpoints',{}).pop(name,None)
        save_bundle(afpo, checkpoint, state)
        pending=[output for output in state['order'] if output not in state['components']]
        if result.get('stopped') and pending:
            return {'checkpoint':str(checkpoint),'manifest':state['manifest'],
                    'generation':result['generation'],'complete':False,'pending_outputs':pending,
                    'selected':f'Saved {name}; remaining output searches need resume',
                    'equation':f'Completed outputs: {", ".join(state["components"])}'}
    model, complexity = bundle_model(afpo, state)
    afpo.SEQUENCE_LAYOUT = state['maps'].get(afpo.SEQUENCE_LAYOUT_KEY)
    metrics = {}
    def score(label, X, Y):
        if X is None:
            return
        prediction = afpo.predict_targets(model, X, state['cats'])
        metrics[label] = {name: ({'rmse': float(np.sqrt(np.mean((prediction[:,i]-Y[:,i])**2)))}
                                 if state['cats'][i] is None else
                                 {'accuracy': float(np.mean(prediction[:,i] == Y[:,i]))})
                          for i,name in enumerate(state['out_names'])}
    score('validation', state['Xv'], state['Yv'])
    if state['args'].get('test_csv'):
        test = afpo.read_dataset(state['args']['test_csv'], state['setup']['delimiter'],
                                 state['args']['max_rows'], state['sample_seed'])
        Xtest,Ytest,_,_,_,_ = afpo.encode(test,state['setup']['types'],state['maps'])
        score('test', Xtest, Ytest)
    # Validation is on local trees; trusted stage ADFs are never expanded for
    # structural validation or evolutionary limits.
    afpo.export_model(model, state['names'], state['out_names'], state['cats'], state['maps'],
                      list(state['fixture'].columns), state['setup']['types'], state['fixture'], state['ranges'])
    if state['args'].get('symbolic_export','on') == 'on':
        afpo.write_symbolic_export(model,state['names'],state['out_names'],state['cats'],state['Xt'])
    card = Path(state['manifest']).with_name('model_card.json')
    card.write_text(json.dumps({'format_version': 1, 'search_mode': 'independent_outputs',
                               'input_relations': state['args']['input_relation'],
                               'output_relations': state['args']['output_relation'],
                               'complexity': complexity, 'metrics': metrics,
                               'formulae': afpo.equations(model,state['names'],state['out_names'],state['cats']),
                               'stage_definitions': afpo.adf_display_definitions(model,state['names']),
                               'outputs': {name: {k:v for k,v in item.items() if k != 'model'}
                                           for name,item in state['components'].items()}}, indent=2))
    print('Per-output complexity: ' + ', '.join(f'{name}: {item["local_nodes"]} local, {item["expanded_nodes"]} expanded'
                                                  for name,item in complexity['per_output'].items()))
    print(f'Total expanded complexity: {complexity["total_expanded_nodes"]}')
    return {'checkpoint': str(checkpoint), 'manifest': state['manifest'], 'model_card': str(card),
            'generation': max(c['generation'] for c in state['components'].values()),
            'complete':True, 'selected': 'independent outputs', 'equation': afpo.equations(model,state['names'],state['out_names'],state['cats'])}


def recover_finished_component(afpo, state):
    """Recover a finalized selection omitted by older Stop handling.

    This only changes the decoded state in memory. Periodic checkpoints without
    a finalized selection are never promoted to completed outputs.
    """
    from dataclasses import asdict
    checkpoint=state.get('active_checkpoint')
    if not checkpoint:
        return False
    generation,population,_,archive,saved=afpo.load_checkpoint(checkpoint)
    outputs=saved.get('out_names',[])
    if len(outputs)!=1 or outputs[0] not in state['order'] or not saved.get('selection'):
        return False
    name=outputs[0]
    if name in state['components']:
        return False
    parents=state['parents'][name]
    if any(parent not in state['components'] for parent in parents):
        return False
    expected_names=[*state['names'],*[f'predicted:{parent}:{head}' for parent in parents
                    for head in range(len(state['components'][parent]['model']['trees']))]]
    if saved['names'] != expected_names:
        return False
    selected=saved.get('selected_model')
    if selected is None:
        # Older checkpoints retain the selected model in their persistent
        # archives/banks. Require an exact match to the final model card;
        # recovering an arbitrary best candidate would lose the user's choice.
        card=Path(saved.get('manifest',checkpoint)).with_name('model_card.json')
        if not card.is_file():
            return False
        evidence=json.loads(card.read_text())
        formulae=evidence.get('formulae')
        if not formulae or not evidence.get('mdl'):

            return False
        candidates=[*population,*archive.items]
        afpo._walk_model_data(saved,lambda data: candidates.append(afpo.Model(**data)))
        matches=[]
        for model in candidates:
            try:
                if (afpo.equations(model,saved['names'],outputs,saved['cats'])==formulae
                        and afpo.model_description(model,len(saved['names']))==evidence['mdl']):
                    matches.append(model)
            except (ValueError,IndexError,KeyError):
                continue
        if not matches:
            return False
        selected=asdict(matches[0])
    state['components'][name]={'model':selected,'parents':parents,
                               'checkpoint':str(checkpoint),'generation':generation}
    state['active_checkpoint']=None
    return True


def load_bundle_state(afpo, path):
    with Path(path).open('rb') as source:
        if b'"afpo-independent-outputs-v1"' not in source.read(128):
            return None
    payload=json.loads(Path(path).read_text())
    if payload.get('format') != 'afpo-independent-outputs-v1':
        return None
    state=afpo._from_json_checkpoint_value(payload['state'])
    recover_finished_component(afpo,state)
    return state


def resume_bundle(afpo, args):
    if getattr(args, '_output_component', False):
        return False
    state=load_bundle_state(afpo,args.resume)
    if state is None:
        return False
    if len(state['components']) == len(state['order']):
        # Outputs consumed by another search are frozen. Terminal searches
        # can be extended without changing the feature contract of any peer.
        consumed = {p for parents in state['parents'].values() for p in parents}
        state['resume_checkpoints'] = {}
        for name in state['order']:
            component = state['components'][name]
            if name not in consumed and (args.max_generations == 0 or args.max_generations > component['generation']):
                state['resume_checkpoints'][name] = component['checkpoint']
        for name in state['resume_checkpoints']: state['components'].pop(name)
    return run_bundle(afpo, args, state, Path(args.resume),lambda *unused: 0)
