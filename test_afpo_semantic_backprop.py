"""Checks for target-guided subtree mutation and its CLI/GUI controls."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import afpo as a
import afpo_gui as gui
from test_afpo import advance, island


class SemanticBackpropTests(unittest.TestCase):
    def setUp(self):
        self.X=np.linspace(1.,3.,20)[:,None]
        self.ops=["+","-","*","square"]
        self.random_state=a.rng.getstate()
        self.numpy_state=np.random.get_state()
        settings=("LOSS_NOISE_FLOOR","FIT_BACKEND","EQUIVALENCE_COLLAPSE","RESIDUAL_ARCHIVE","QD_PARENT_CHOICE",
                  "SCALE_BALANCED_SELECTION","GUARD_EXPLOIT_CHECK","INTERPOLATION_CHECK","JUMP_CONSTANT_SCAN",
                  "SELECTION_PROBE_FILTER","JUMP_MUTATION_WEIGHT","SQUASH_SWAP_WEIGHT","SMOOTH_SWAP_WEIGHT",
                  "GATE_MUTATION_WEIGHT","SEQUENCE_LAYOUT","SEQUENCE_GROUP_REQUEST","_SELECTION_PROBES")
        for name in settings: self.addCleanup(setattr,a,name,getattr(a,name))
        a.rng.seed(5)

    def tearDown(self):
        a.rng.setstate(self.random_state)
        np.random.set_state(self.numpy_state)

    def test_inversion_preserves_siblings_and_signed_branches(self):
        examples=[
            (("+",("x",0),("c",2.)),(0,),self.X[:,0]+4.,self.X[:,0]+2.),
            (("-",("c",2.),("x",0)),(1,),2.-self.X[:,0]*2.,self.X[:,0]*2.),
            (("*",("c",3.),("x",0)),(1,),self.X[:,0]*6.,self.X[:,0]*2.),
            (("/",("c",6.),("x",0)),(1,),3./self.X[:,0],self.X[:,0]*2.),
            (("sqrt",("x",0)),(0,),-self.X[:,0],-self.X[:,0]**2),
            (("square",("neg",("x",0))),(0,),self.X[:,0]**2,-self.X[:,0]),
            (("inv",("neg",("x",0))),(0,),1./(self.X[:,0]+a.EPS),-self.X[:,0]),
        ]
        for tree,path,desired,expected in examples:
            with self.subTest(tree=tree):
                np.testing.assert_allclose(a.desired_subtree_semantics(tree,path,self.X,desired),expected)

    def test_unreachable_and_unsupported_inverses(self):
        zero=("*",("x",0),("c",0.))
        self.assertTrue(np.isnan(a.desired_subtree_semantics(zero,(0,),self.X,np.ones(20))).all())
        self.assertIsNone(a.desired_subtree_semantics(("sin",("x",0)),(0,),self.X,np.ones(20)))
        self.assertTrue(np.isnan(a.desired_subtree_semantics(("square",("x",0)),(0,),self.X,-np.ones(20))).all())

    def test_generic_sine_chooses_nearest_branch_and_refines_touch(self):
        current=np.array([.2,3.,6.5,-3.])
        solved=a.numerical_child_target("sin",[current],0,np.full(4,.5))
        np.testing.assert_allclose(solved,[np.pi/6,5*np.pi/6,2*np.pi+np.pi/6,-7*np.pi/6],atol=1e-6)
        touch=a.numerical_child_target("sin",[np.array([1.,-4.4])],0,np.ones(2))
        np.testing.assert_allclose(np.sin(touch),1.,atol=1e-6)
        np.testing.assert_allclose(touch,[np.pi/2,-3*np.pi/2],atol=2e-3)
        impossible=a.numerical_child_target("sin",[current],0,np.full(4,2.))
        self.assertTrue(np.isnan(impossible).all())
        far=np.array([1e6,-1e6])
        roots=np.array([np.pi/6,5*np.pi/6])[:,None]
        branches=roots+2*np.pi*np.rint((far-roots)/(2*np.pi))
        closest=branches[np.argmin(np.abs(branches-far),axis=0),np.arange(2)]
        solved=a.numerical_child_target("sin",[far],0,np.full(2,.5))
        np.testing.assert_allclose(solved,closest,rtol=0.,atol=1e-6)

    def test_generic_discontinuities_require_verified_solutions(self):
        values=[np.array([.9,2.9,-.1]),np.ones(3)]
        invalid=a.numerical_child_target("floordiv",values,0,np.full(3,.5))
        self.assertTrue(np.isnan(invalid).all())
        for op,target in (("floordiv",1.),("mod",.1)):
            solved=a.numerical_child_target(op,values,0,np.full(3,target))
            self.assertTrue(np.isfinite(solved).all())
            np.testing.assert_allclose(a.fast_op_eval(op,[solved,values[1]]),target,atol=1e-6)

    def test_generic_square_and_pow_handle_multiple_children(self):
        current=np.array([-2.,2.,0.])
        square=a.numerical_child_target("square",[current],0,np.full(3,9.))
        np.testing.assert_allclose(np.abs(square),3.,atol=1e-6)
        self.assertLess(square[0],0.); self.assertGreater(square[1],0.)
        power=a.numerical_child_target("pow",[np.full(3,2.),np.array([1.,2.,3.])],1,np.full(3,16.))
        np.testing.assert_allclose(power,4.,atol=1e-6)

    def test_generic_reaches_under_sine_and_adf(self):
        tree=("sin",("+",("x",0),("c",.1)))
        desired=np.sin(self.X[:,0]+.2)
        self.assertIsNone(a.desired_subtree_semantics(tree,(0,1),self.X,desired))
        wanted=a.desired_subtree_semantics(tree,(0,1),self.X,desired,mode="generic")
        np.testing.assert_allclose(np.sin(self.X[:,0]+wanted),desired,atol=1e-6)
        adfs={"adf_0":{"tree":("sin",("arg",0)),"arity":1}}
        wanted=a.desired_subtree_semantics(("adf_0",("x",0)),(0,),self.X,desired,adfs,mode="generic")
        np.testing.assert_allclose(np.sin(wanted),desired,atol=1e-6)

    def test_generic_library_includes_all_inputs_unary_and_capped_binary(self):
        library=a.semantic_backprop_library(3,["sin","pow","+"],[("c",7.)])
        for feature in range(3):
            self.assertIn(("x",feature),library)
            self.assertIn(("sin",("x",feature)),library)
        self.assertIn(("pow",("x",2),("x",1)),library)
        self.assertIn(("c",7.),library)
        big=a.semantic_backprop_library(50,["+","*","sin"])
        self.assertEqual(sum(node[0] in ("+","*") for node in big),2000)

    def test_generic_drops_outliers_requires_half_rows_and_improves_subtree(self):
        targets=np.r_[np.linspace(1,2,19),1e10]
        mask=a.semantic_backprop_mask(targets)
        self.assertEqual(int(mask.sum()),19); self.assertFalse(mask[-1])
        X=self.X*.2
        tree=("sin",("x",0)); desired=np.sin(.5*X[:,0])
        with patch.object(a.rng,"choice",return_value=(0,)):
            child=a.semantic_backprop_mutate(tree,X,desired,1,["+","*","sin"],15,4,mode="generic")
        self.assertNotEqual(tree,child)
        np.testing.assert_allclose(a.evaluate(child,X),desired,atol=1e-6)
        with patch.object(a,"desired_subtree_semantics",return_value=np.r_[np.ones(9),np.full(11,np.nan)]):
            self.assertEqual(a.semantic_backprop_mutate(tree,X,desired,1,self.ops,15,4,mode="generic"),tree)
        with patch.object(a.rng,"choice",return_value=()):
            self.assertEqual(a.semantic_backprop_mutate(tree,self.X,a.evaluate(tree,self.X),1,["+","*","sin"],15,4,mode="generic"),tree)

    def test_directed_mutation_improves_root_within_limits(self):
        tree=("+",("square",("x",0)),("x",0))
        desired=self.X[:,0]**2+3.*self.X[:,0]+2.
        with patch.object(a.rng,"choice",return_value=(1,)):
            child=a.semantic_backprop_mutate(tree,self.X,desired,1,self.ops,15,4)
        self.assertNotEqual(child,tree)
        np.testing.assert_allclose(a.evaluate(child,self.X),desired,atol=1e-10)
        self.assertLessEqual(a.node_size(child),15)
        self.assertLessEqual(a.node_depth(child),4)
        self.assertIn(a.subtree_at(tree,(0,)),list(a.walk_tree(child)))

    def test_regression_targets_undo_affine_and_handle_zero_scale(self):
        model=a.Model([("x",0)],[(2.,3.)])
        target=4.*self.X+5.
        np.testing.assert_allclose(a.semantic_backprop_targets(model,self.X,target,[None])[0],2.*self.X[:,0]+1.)
        model.scales=[(0.,3.)]
        self.assertEqual(a.semantic_backprop_targets(model,self.X,target,[None]),[None])

    def test_classification_targets_follow_cross_entropy_gradient(self):
        binary=a.Model([("c",.5)],[(1.,0.)])
        truth=np.ones((20,1))
        np.testing.assert_allclose(a.semantic_backprop_targets(binary,self.X,truth,[["no","yes"]])[0],1.)
        multi=a.Model([("c",0.)]*3,[(1.,0.)]*3)
        targets=a.semantic_backprop_targets(multi,self.X,truth,[["a","b","c"]])
        np.testing.assert_allclose(np.column_stack(targets),np.tile([-1./3,2./3,-1./3],(20,1)))

    def test_cli_and_gui_default_and_explicit_choices(self):
        self.assertEqual(a.parse_cli([])[1].semantic_backprop,"on")
        option=next(item for item in gui.options()["advanced"] if item["dest"]=="semantic_backprop")
        self.assertEqual((option["kind"],option["default"],option["choices"]),("choice","on",["on","off"]))
        for value in ("on","off"):
            args=a.parse_cli(gui.build_argv({"advanced":{"semantic_backprop":value}}))[1]
            self.assertEqual(args.semantic_backprop,value)
            self.assertTrue(args.semantic_backprop_explicit)
        option=next(item for item in gui.options()["advanced"] if item["dest"]=="semantic_backprop_mode")
        self.assertEqual((option["default"],option["choices"]),("exact",["exact","generic"]))
        for value in ("exact","generic"):
            args=a.parse_cli(gui.build_argv({"advanced":{"semantic_backprop_mode":value}}))[1]
            self.assertEqual(args.semantic_backprop_mode,value)
            self.assertTrue(args.semantic_backprop_mode_explicit)

    def test_evolution_toggle_reaches_operator(self):
        Y=self.X**2
        for enabled,mode in ((False,"exact"),(True,"exact"),(True,"generic")):
            with self.subTest(enabled=enabled,mode=mode):
                state=island(self.X,[None],self.ops)
                evaluator=a.ModelEvaluator(1,{"train":(self.X,Y)},False,[None],a.compile_constraints(),["y"])
                try:
                    with patch.object(a,"semantic_backprop_mutate",wraps=a.semantic_backprop_mutate) as directed:
                        for generation in range(3):
                            advance(state,generation,self.X,Y,[None],self.ops,evaluator,semantic_backprop=enabled,semantic_backprop_mode=mode,bayesian_mode="off")
                        self.assertEqual(directed.called,enabled)
                        if enabled: self.assertTrue(all(call.kwargs["mode"]==mode for call in directed.call_args_list))
                finally:
                    evaluator.close()

    def test_checkpoint_preserves_toggle_and_accepts_resume_override(self):
        frame=pd.DataFrame({"x":self.X[:,0],"y":self.X[:,0]**2})
        with tempfile.TemporaryDirectory(prefix="afpo-backprop-") as directory,contextlib.chdir(directory):
            frame.to_csv("data.csv",index=False)
            args=a.parse_cli(["--population","8","--max-generations","1","--seed","5","--workers","1",
                              "--fit-backend","python","--semantic-backprop","off","--semantic-backprop-mode","generic"])[1]
            setup={"path":Path("data.csv"),"df":frame,"types":[1,5],"delimiter":",","ops":self.ops,
                   "affine_on":True,"coev":False,"dynamic_pressure_on":False,"adf_enabled":False,"nodes":15,"depth":4,
                   "island_count":1,"migration_interval":5,"migrants_per_island":1,"val_path":"0","validation_percent":None,"metadata":{}}
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=a.train_from_setup(args,setup,choose_model=lambda *_:0)["checkpoint"]
                self.assertFalse(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop"])
                self.assertEqual(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop_mode"],"generic")
                a.resume_main(a.parse_cli(["--resume",checkpoint,"--max-generations","2","--workers","1"])[1])
                self.assertFalse(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop"])
                self.assertEqual(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop_mode"],"generic")
                a.resume_main(a.parse_cli(["--resume",checkpoint,"--max-generations","3","--workers","1",
                                          "--semantic-backprop=on","--semantic-backprop-mode=exact"])[1])
                self.assertTrue(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop"])
                self.assertEqual(a.load_checkpoint(checkpoint,False)[4]["semantic_backprop_mode"],"exact")


if __name__=="__main__":
    unittest.main()
