"""Reproduce the tuple-loss failure through the production smoke dispatch."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import train_refiner as runner
from scripts import wave2_smoke as smoke
from tests.test_wave_two import Tiles, models


class SmokeEntrypointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_tuple_loss_regression_through_production_entrypoint(self):
        logits=torch.randn(2,1,32,32,requires_grad=True)
        target=torch.zeros(2,32,32)
        valid=torch.ones_like(target,dtype=torch.bool)
        returned=runner.deep_supervision([logits],target,valid)
        self.assertIsInstance(returned,tuple)
        with self.assertRaises(TypeError):
            torch.isfinite(returned)  # The exact expression that failed in job 8772834.
        encoder,head=models()
        original_model=runner.LoopRefiner
        datasets={domain:Tiles("Val",domain) for domain in runner.DOMAINS}
        groups={domain:{key:{"scores":{"A":.5,"quality":.5}} for key in ("y0_draft","1","2","4","8")}
                for domain in (*runner.DOMAINS,"Overall")}
        images=[dict(domain=domain,image_id=f"{domain}{i}") for domain in runner.DOMAINS for i in range(2)]
        cache=type("SyntheticCache",(),{"load":lambda self,ids,device:torch.zeros(len(ids),1,32,32)})()

        def production_process(command,**kwargs):
            with patch.object(sys,"argv",command[2:]),redirect_stdout(kwargs["stdout"]), \
                 patch.object(runner.y0,"load_frozen_probe",return_value=(copy.deepcopy(encoder),copy.deepcopy(head),{"head":"conv_up"})), \
                 patch.object(runner,"LoopRefiner",side_effect=lambda *a,**kw:original_model(8,dimension=16,heads=4,patch=4,image_size=32)), \
                 patch.object(runner.probe,"RoadDataset",side_effect=Tiles), \
                 patch.object(runner,"protocol_datasets",return_value=datasets), \
                 patch.object(runner,"DraftCache",return_value=cache), \
                 patch.object(runner,"evaluate",return_value={"groups":groups,"images":images,"wall_seconds":0.}), \
                 patch.object(runner,"summary_markdown",return_value="# Synthetic fixture"):
                runner.main()
            return subprocess.CompletedProcess(command,0)

        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as temporary:
            out=Path(temporary)
            command=smoke.stage_commands(out/"cache",out)["RS"]
            command[command.index("--device")+1]="cpu"
            with patch.object(smoke.subprocess,"run",side_effect=production_process):
                result=smoke.run_stage("RS",command,out)
            self.assertTrue(result["passed"],result)
            self.assertEqual(result["updates"],20)
            data=json.loads((out/"RS/report/metrics.json").read_text())
            self.assertEqual(data["config"]["max_steps"],20)

    def test_one_failure_still_executes_all_four_production_commands(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/"runs") as temporary:
            out=Path(temporary)
            commands=smoke.stage_commands(out/"cache",out)
            called=[]
            def process(command,**kwargs):
                arm=next(arm for arm,value in commands.items() if command==value)
                called.append(arm)
                if arm=="RS":
                    kwargs["stdout"].write("Synthetic RS failure\n")
                    return subprocess.CompletedProcess(command,1)
                path=out/arm/("report/metrics.json" if arm=="RW" else "metrics.json")
                path.parent.mkdir(parents=True)
                images=[dict(domain=d,image_id=f"{d}{i}") for d in runner.DOMAINS for i in range(2)]
                path.write_text(json.dumps(dict(epochs=[dict(loss=.5,batches=[dict(loss=.5)]*20)],
                                               evaluation=dict(images=images),frozen_head_unchanged=True)))
                return subprocess.CompletedProcess(command,0)
            report=dict(cache=dict(gates_passed=True,passed=True,exit_code=0))
            output=io.StringIO()
            with patch.object(smoke.subprocess,"run",side_effect=process),redirect_stdout(output):
                smoke.run_stages(commands,out,report)
                success=smoke.finish(report)
            self.assertEqual(called,["RS","RW","LA","LC"])
            self.assertFalse(success)
            self.assertIn("WAVE2_SMOKE_FAILED",output.getvalue())
            self.assertNotIn("WAVE2_SMOKE_COMPLETE",output.getvalue())
            self.assertEqual(report["RS"]["exit_code"],1)
            self.assertTrue(all(report[arm]["passed"] for arm in ("RW","LA","LC")))
            self.assertTrue(report["LA"]["head_and_BN_unchanged"])
            for arm,command in commands.items():
                script="scripts/train_refiner.py" if arm in ("RS","RW") else "scripts/train_lora_draft.py"
                self.assertEqual(command[2],script)
                self.assertEqual(command[command.index("--max-steps")+1],"20")
                self.assertEqual(command[command.index("--limit-per-domain")+1],"2")

    def test_max_steps_repeats_small_loader_and_default_is_one_pass(self):
        self.assertEqual(list(runner.training_batches([1,2],5)),[1,2,1,2,1])
        self.assertEqual(list(runner.training_batches([1,2])),[1,2])
