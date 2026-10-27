import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from validate_readiness import GATES, validate

class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / 'evidence.txt').write_text('synthetic fixture; no scientific truth claimed')
        self.evidence = {'path': 'evidence.txt', 'sha256': hashlib.sha256((self.root / 'evidence.txt').read_bytes()).hexdigest(), 'subject_revision': 'fixture-revision'}
        self.record = {'domain': 'general', 'mathematical_claims': True, 'schema_version': 2, 'kind': 'project_record', 'task_state': 'complete', 'deliverable_kind': 'proof', 'subject_revision': 'fixture-revision', 'completion_criterion': 'Fixture schema check', 'blockers': [], 'active_operations': [], 'ownership': {'integrator': None, 'assignments': []}, 'gates': [{'id': key, 'applicable': True, 'outcome': 'passed', 'evidence': [copy.deepcopy(self.evidence)]} for key in sorted(GATES)]}
    def rejected(self, change):
        record = copy.deepcopy(self.record)
        change(record)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validate(record, self.root, True)
    def test_cancelled(self):
        self.rejected(lambda r: r.update(task_state='cancelled'))
    def test_bad_assignment(self):
        self.rejected(lambda r: r['ownership'].update(assignments=[None]))
    def test_bad_gate_id(self):
        self.rejected(lambda r: r['gates'][0].update(id=[]))
    def test_cymatica_integrity(self):
        def change(r):
            r.update(domain='cymatica', deliverable_kind='software', mathematical_claims=False)
            next(g for g in r['gates'] if g['id']=='mathematical_integrity').update(applicable=False, outcome='not_applicable', applicability_review={'reason':'avoid', 'reviewer':'fixture'})
        self.rejected(change)
    def test_foundation_always(self):
        def change(r):
            r.update(domain='general', deliverable_kind='documentation', mathematical_claims=False)
            next(g for g in r['gates'] if g['id']=='foundation_traceability').update(applicable=False, outcome='not_applicable', applicability_review={'reason':'avoid', 'reviewer':'fixture'})
        self.rejected(change)
    def test_distinct_source_escape(self):
        source = self.root / 'source'
        evidence = self.root / 'logs'
        source.mkdir()
        evidence.mkdir()
        (evidence / 'evidence.txt').write_bytes((self.root / 'evidence.txt').read_bytes())
        (source / 'outside').symlink_to(evidence, target_is_directory=True)
        self.record['ownership'].update(integrator='i', assignments=[{'owner':'a', 'write_paths':['outside/file']}])
        with self.assertRaises(ValueError):
            validate(self.record, evidence, True, source)
    def test_distinct_source_alias(self):
        source = self.root / 'source'
        evidence = self.root / 'logs'
        source.mkdir()
        evidence.mkdir()
        (evidence / 'evidence.txt').write_bytes((self.root / 'evidence.txt').read_bytes())
        (source / 'src').mkdir()
        (source / 'alias').symlink_to(source / 'src', target_is_directory=True)
        self.record['ownership'].update(integrator='i', assignments=[{'owner':'a', 'write_paths':['src']}, {'owner':'b', 'write_paths':['alias/file']}])
        with self.assertRaises(ValueError):
            validate(self.record, evidence, True, source)
    def test_valid(self):
        self.assertEqual(validate(self.record, self.root, True)['readiness_attestation'], 'consistent')
    def test_missing_gate(self):
        self.rejected(lambda r: r['gates'].pop())
    def test_unknown_gate(self):
        self.rejected(lambda r: r['gates'][0].update(id='unrelated'))
    def test_missing_evidence(self):
        self.rejected(lambda r: r['gates'][0]['evidence'][0].update(path='absent'))
    def test_hash(self):
        self.rejected(lambda r: r['gates'][0]['evidence'][0].update(sha256='0'*64))
    def test_revision(self):
        self.rejected(lambda r: r['gates'][0]['evidence'][0].update(subject_revision='stale'))
    def test_unresolved_applicability(self):
        self.rejected(lambda r: r['gates'][0].update(applicable=None))
    def test_unreviewed_na(self):
        self.rejected(lambda r: r['gates'][0].update(applicable=False, outcome='not_applicable'))
    def test_profile_na(self):
        self.rejected(lambda r: next(g for g in r['gates'] if g['id']=='mathematical_integrity').update(applicable=False, outcome='not_applicable', applicability_review={'reason':'avoid test', 'reviewer':'fixture'}))
    def test_legitimate_na(self):
        for gate in self.record['gates']:
            if gate['id'] in {'actual_runtime_validation', 'exact_unpacked_deliverable', 'remote_revision_and_ci', 'authorization_before_publication'}:
                gate.update(applicable=False, outcome='not_applicable', applicability_review={'reason':'Local proof-document fixture, no software or release', 'reviewer':'fixture'})
        validate(self.record, self.root, True)
    def test_parent_overlap(self):
        self.rejected(lambda r: r['ownership'].update(integrator='i', assignments=[{'owner':'a', 'write_paths':['src']}, {'owner':'b', 'write_paths':['src/module.py']}]))
    def test_alias_overlap(self):
        self.rejected(lambda r: r['ownership'].update(integrator='i', assignments=[{'owner':'a', 'write_paths':['src/file']}, {'owner':'b', 'write_paths':['src/./file']}]))
    def test_traversal(self):
        self.rejected(lambda r: r['gates'][0]['evidence'][0].update(path='../evidence.txt'))
    def test_symlink_escape(self):
        (self.root / 'escape').symlink_to('/etc/passwd')
        self.rejected(lambda r: r['gates'][0]['evidence'][0].update(path='escape'))
    def test_template_completion(self):
        self.rejected(lambda r: r.update(kind='template'))
    def test_active_operations(self):
        self.rejected(lambda r: r.update(active_operations=['running']))
    def test_failed_gate(self):
        self.rejected(lambda r: r['gates'][0].update(outcome='failed'))
    def test_unknown_revision(self):
        self.rejected(lambda r: r.update(subject_revision=None))

if __name__ == '__main__':
    unittest.main()
