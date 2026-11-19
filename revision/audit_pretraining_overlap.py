"""Aggregate-only patient overlap audit. No patient IDs are written or printed.

Exit: 0 verified zero overlap; 2 overlap; 3 incomplete/unverifiable inputs.
Patient namespaces and frozen source coverage must be declared explicitly.
"""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np,pandas as pd

EXPECTED={'MHI','CODE-15','MIMIC-IV'}

class AuditInputError(ValueError):
    pass


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_columns(path,columns):
    p=Path(path)
    if p.suffix=='.parquet':return pd.read_parquet(p,columns=columns)
    if p.suffix=='.csv':return pd.read_csv(p,usecols=columns,dtype='string',keep_default_na=False)
    raise AuditInputError('Unsupported input format')


def canonical(series,kind):
    if pd.api.types.is_float_dtype(series.dtype):
        raise AuditInputError('Floating-point patient IDs are prohibited')
    if series.isna().any():raise AuditInputError('Missing patient IDs')
    if any(isinstance(v,(float,np.floating,bool,np.bool_)) for v in series):
        raise AuditInputError('Floating-point or boolean patient IDs are prohibited')
    s=series.astype(str).str.strip()
    if s.eq('').any():raise AuditInputError('Empty patient IDs')
    if s.str.lower().isin(['nan','none','null','na','n/a','<na>']).any():
        raise AuditInputError('Missing-value sentinel in patient IDs')
    if kind=='integer':
        if not s.str.fullmatch(r'[0-9]+').all():raise AuditInputError('Non-integer patient IDs in integer mode')
        return s.map(lambda v:str(int(v)))
    if kind=='string':
        if s.str.fullmatch(r'[0-9]+\.[0-9]+|[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+').any():
            raise AuditInputError('Numeric float/exponent serialization in string IDs')
        return s
    raise AuditInputError('Unknown patient-ID encoding')


def load_ids(spec):
    path=Path(spec['path']);col=spec['patient_column']
    frame=read_columns(path,[col]);ids=canonical(frame[col],spec['id_encoding'])
    report={'path':str(path),'sha256':checksum(path),'rows':len(ids),'unique_patients':int(ids.nunique()),'namespace':spec['namespace'],'coverage':spec.get('coverage','undeclared')}
    if report['coverage']!='all_pretraining_splits' and spec.get('role')!='validation':
        raise AuditInputError('Source must declare coverage of every pretraining split')
    return ids,report


def map_namespace(ids,spec,target_namespace,target_encoding):
    if spec['namespace']==target_namespace:
        if spec['id_encoding']!=target_encoding:
            raise AuditInputError('Conflicting encodings in the same patient namespace')
        return ids,None
    cw=spec.get('crosswalk')
    if not cw:return None,{'status':'not_comparable','reason':'No verified patient crosswalk to the validation namespace'}
    if cw.get('source_namespace')!=spec['namespace'] or cw.get('target_namespace')!=target_namespace:
        raise AuditInputError('Crosswalk namespace mismatch')
    if cw.get('target_id_encoding')!=target_encoding:
        raise AuditInputError('Crosswalk target encoding differs from validation encoding')
    frame=read_columns(cw['path'],[cw['source_column'],cw['target_column']])
    left=canonical(frame[cw['source_column']],spec['id_encoding']);right=canonical(frame[cw['target_column']],cw['target_id_encoding'])
    pairs=pd.DataFrame({'source':left,'target':right}).drop_duplicates()
    if pairs.source.duplicated().any():raise AuditInputError('Ambiguous patient crosswalk')
    mapping=pairs.set_index('source').target;mapped=ids.map(mapping)
    if mapped.isna().any():
        return None,{'status':'incomplete_linkage','unmapped_manifest_rows':int(mapped.isna().sum()),'crosswalk_sha256':checksum(cw['path'])}
    return mapped,{'status':'complete_linkage','crosswalk_sha256':checksum(cw['path']),'source_namespace':spec['namespace'],'target_namespace':target_namespace}


def audit(config):
    if set(config.get('expected_pretraining_sources',[]))!=EXPECTED:
        raise AuditInputError('Expected source list must include MHI, CODE-15 and MIMIC-IV')
    if not config.get('encoder_sha256') or not re.fullmatch(r'[a-f0-9]{64}',config['encoder_sha256']):
        raise AuditInputError('A frozen encoder SHA-256 is required')
    validation=dict(config['validation'],role='validation')
    vids,vreport=load_ids(validation)
    if vids.duplicated().any():raise AuditInputError('Validation manifest must have one row per unique patient')
    if len(vids)!=config.get('expected_validation_n',21647):raise AuditInputError('Validation denominator mismatch')
    vset=set(vids);rows=[];coverage=set();overlap_union=set();unverifiable=False
    for spec in config['pretraining']:
        if spec['source'] not in EXPECTED:raise AuditInputError('Undeclared pretraining source')
        coverage.add(spec['source']);ids,report=load_ids(spec);mapped,link=map_namespace(ids,spec,validation['namespace'],validation['id_encoding'])
        report.update(source=spec['source'],linkage=link)
        if mapped is None:
            unverifiable=True;report['overlap_patients']=None
        else:
            shared=set(mapped)&vset;overlap_union.update(shared);report['overlap_patients']=len(shared)
        rows.append(report)
    missing=sorted(EXPECTED-coverage)
    status='overlap_found' if overlap_union else 'inconclusive' if missing or unverifiable else 'verified_zero_overlap'
    return {'status':status,'encoder_sha256':config['encoder_sha256'],'validation':vreport,'pretraining_sources':rows,
        'scope':'Comparison of the declared input manifests only; the data custodian must independently establish their provenance and completeness for this encoder before claiming study independence.',
        'missing_sources':missing,'unique_validation_patients_overlapping_comparable_sources':len(overlap_union),
        'limitations':'Zero among comparable sources is not an overall independence claim when any source or linkage is incomplete. Exact namespaces must be justified by the data custodian. Manifests must contain all splits actually used by the frozen encoder. No patient IDs are exported.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    try:
        with Path(a.config).open() as f:c=json.load(f)
        result=audit(c)
    except (AuditInputError,KeyError,FileNotFoundError,OSError,ValueError,TypeError) as e:
        # Do not expose unexpected input values or row-level identifiers in exceptions.
        result={'status':'inconclusive','error_class':type(e).__name__,'reason':'Input, namespace, ID integrity, denominator or coverage validation failed; inspect the approved input configuration privately.'}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'report':str(out)}))
    return {'verified_zero_overlap':0,'overlap_found':2,'inconclusive':3}[result['status']]

if __name__=='__main__':raise SystemExit(main())
