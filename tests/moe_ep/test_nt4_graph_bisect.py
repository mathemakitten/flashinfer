import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bisect_graph', Path(__file__).parents[2] / 'examples/moe_ep/nt4_graph_bisect.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Graph:
    def __init__(self): self.events=[]
    def capture_begin(self,*a,**kw): self.events.append(('begin',kw))
    def capture_end(self): self.events.append(('end',))
    def register_generator_state(self,g): self.events.append(('rng',g))
    def pool(self): return (1,2)
    def reset(self): self.events.append(('reset',))
    def replay(self): self.events.append(('replay',))

class Tests(unittest.TestCase):
    def test_discovers_actual_stack_type_and_subclasses(self):
        class HybridStack:
            pass
        class WrappedStack(HybridStack):
            pass
        stack = WrappedStack()
        model = SimpleNamespace(modules=lambda: iter([object(), stack]))
        self.assertIs(m.find_stack(model, HybridStack), stack)
        for modules in ([], [HybridStack(), HybridStack()]):
            with self.assertRaisesRegex(RuntimeError, "needs one HybridStack"):
                m.find_stack(SimpleNamespace(modules=lambda: iter(modules)), HybridStack)

    def test_segments_preserve_pool_rng_and_boundary(self):
        session=SimpleNamespace(segment_split=20,forward_index=1)
        sync=[]
        cuda=SimpleNamespace(CUDAGraph=Graph,synchronize=lambda:sync.append(1))
        with patch.dict(sys.modules,torch=SimpleNamespace(cuda=cuda)):
            g=m.SegmentedGraph(session)
            g.register_generator_state('rng')
            g.capture_begin(capture_error_mode='thread_local')
            value=object();g.split(value);g.capture_end()
            self.assertIs(g.retained[0],value)
            self.assertIn(('rng','rng'),g.graphs[1].events)
            self.assertEqual(g.graphs[1].events[-2][1]['pool'],(1,2))
            self.assertIsNone(session.segment_active)
            g.replay();self.assertEqual(len(sync),4)
            g.reset();self.assertFalse(g.retained)

    def test_missing_boundary_fails(self):
        with patch.dict(sys.modules,torch=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))):
            g=m.SegmentedGraph(SimpleNamespace());g.capture_begin()
            with self.assertRaisesRegex(RuntimeError,'not reached'):g.capture_end()

if __name__=='__main__':unittest.main()
