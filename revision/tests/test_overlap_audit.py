import copy
import pandas as pd,numpy as np,pytest
from audit_pretraining_overlap import audit,canonical,AuditInputError


def config(tmp_path):
    def write(name,values):
        p=tmp_path/(name+'.parquet');pd.DataFrame({'patient':values}).to_parquet(p,index=False);return str(p)
    v=write('validation',[1,2,3])
    sources=[]
    for name in ['MHI','CODE-15','MIMIC-IV']:
        sources.append({'source':name,'path':write(name,[10,11]),'patient_column':'patient','namespace':'canonical_patient','id_encoding':'integer','coverage':'all_pretraining_splits'})
    return {'encoder_sha256':'a'*64,'expected_pretraining_sources':['MHI','CODE-15','MIMIC-IV'],'expected_validation_n':3,'validation':{'path':v,'patient_column':'patient','namespace':'canonical_patient','id_encoding':'integer'},'pretraining':sources}


def test_zero_requires_every_source_and_comparable_namespaces(tmp_path):
    c=config(tmp_path);assert audit(c)['status']=='verified_zero_overlap'
    c['pretraining'].pop();assert audit(c)['status']=='inconclusive'
    c=config(tmp_path);c['pretraining'][1]['namespace']='code15_local'
    a=audit(c);assert a['status']=='inconclusive' and a['pretraining_sources'][1]['overlap_patients'] is None


def test_overlap_and_duplicate_ecgs_are_counted_by_patient(tmp_path):
    c=config(tmp_path);p=c['pretraining'][0]['path'];pd.DataFrame({'patient':[2,2,3,10]}).to_parquet(p,index=False)
    a=audit(c);assert a['status']=='overlap_found';assert a['unique_validation_patients_overlapping_comparable_sources']==2
    assert a['pretraining_sources'][0]['rows']==4 and a['pretraining_sources'][0]['unique_patients']==3


def test_crosswalk_must_be_complete_and_unambiguous(tmp_path):
    c=config(tmp_path);s=c['pretraining'][0];s['namespace']='ecg_index'
    p=tmp_path/'crosswalk.parquet';pd.DataFrame({'local':[10,11],'mrn':[1,20]}).to_parquet(p,index=False)
    s['crosswalk']={'path':str(p),'source_column':'local','target_column':'mrn','source_namespace':'ecg_index','target_namespace':'canonical_patient','target_id_encoding':'integer'}
    assert audit(c)['status']=='overlap_found'
    pd.DataFrame({'local':[10],'mrn':[1]}).to_parquet(p,index=False);assert audit(c)['status']=='inconclusive'
    pd.DataFrame({'local':[10,10,11],'mrn':[1,2,20]}).to_parquet(p,index=False)
    with pytest.raises(AuditInputError):audit(c)


@pytest.mark.parametrize('values',[[1.,2.],[np.float32(16777217)],['123.0'],['1e8'],[''],[None],[True]])
def test_damaged_or_missing_integer_identifiers_rejected(values):
    with pytest.raises(AuditInputError):canonical(pd.Series(values),'integer')


def test_integer_and_string_namespaces_are_not_silently_conflated():
    assert canonical(pd.Series(['001','2']),'integer').tolist()==['1','2']
    assert canonical(pd.Series(['001','2']),'string').tolist()==['001','2']


def test_validation_uniqueness_and_source_split_coverage_required(tmp_path):
    c=config(tmp_path);pd.DataFrame({'patient':[1,1,3]}).to_parquet(c['validation']['path'],index=False)
    with pytest.raises(AuditInputError):audit(c)
    c=config(tmp_path);c['pretraining'][0]['coverage']='train_only'
    with pytest.raises(AuditInputError):audit(c)


def test_namespace_encoding_conflicts_and_string_sentinels_rejected(tmp_path):
    c=config(tmp_path);c['pretraining'][0]['id_encoding']='string'
    with pytest.raises(AuditInputError):audit(c)
    for s in ['nan','NULL','123.0','1e8']:
        with pytest.raises(AuditInputError):canonical(pd.Series([s]),'string')
