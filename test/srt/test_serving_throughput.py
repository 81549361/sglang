import os
import unittest
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, popen_launch_server


class TestServingThroughput(unittest.TestCase):

    def run_test(self, disable_radix_cache, disable_flashinfer, chunked_prefill_size):
        # Launch the server
        other_args = []
        if disable_radix_cache:
            other_args.append("--disable-radix-cache")
        if disable_flashinfer:
            other_args.append("--disable-flashinfer")
        other_args.extend(["--chunked-prefill-size", str(chunked_prefill_size)])

        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = "http://127.0.0.1:9157"
        process = popen_launch_server(
            model, base_url, timeout=300, other_args=other_args
        )

        # Run benchmark
        num_prompts = 400
        args = SimpleNamespace(
            backend="sglang",
            base_url=base_url,
            host=None,
            port=None,
            dataset_name="random",
            dataset_path="",
            model=None,
            tokenizer=None,
            num_prompts=num_prompts,
            sharegpt_output_len=None,
            random_input_len=4096,
            random_output_len=2048,
            random_range_ratio=0.0,
            request_rate=float("inf"),
            multi=None,
            seed=0,
            output_file=None,
            disable_tqdm=False,
            disable_stream=False,
            disable_ignore_eos=False,
            extra_request_body=None,
        )

        try:
            res = run_benchmark(args)
        finally:
            kill_child_process(process.pid)

        assert res["completed"] == num_prompts
        return res

    def test_default(self):
        res = self.run_test(
            disable_radix_cache=ServerArgs.disable_radix_cache,
            disable_flashinfer=ServerArgs.disable_flashinfer,
            chunked_prefill_size=ServerArgs.chunked_prefill_size,
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            # A100 (PCIE) performance
            assert res["output_throughput"] >= 1400

    def test_default_without_radix_cache(self):
        res = self.run_test(
            disable_radix_cache=True,
            disable_flashinfer=ServerArgs.disable_flashinfer,
            chunked_prefill_size=ServerArgs.chunked_prefill_size,
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            # A100 (PCIE) performance
            assert res["output_throughput"] >= 1450

    def test_default_with_chunked_prefill(self):
        res = self.run_test(
            disable_radix_cache=ServerArgs.disable_radix_cache,
            disable_flashinfer=ServerArgs.disable_flashinfer,
            chunked_prefill_size=8192,
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            # A100 (PCIE) performance
            assert res["output_throughput"] >= 1400

    def test_all_cases(self):
        for disable_radix_cache in [False, True]:
            for disable_flashinfer in [False, True]:
                for chunked_prefill_size in [-1, 2048]:
                    self.run_test(
                        disable_radix_cache=False,
                        disable_flashinfer=False,
                        chunked_prefill_size=-1,
                    )


if __name__ == "__main__":
    unittest.main()