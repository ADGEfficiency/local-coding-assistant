import collections
import datasets

import pathlib

data = [
    {
        "instruction": "Complete the code snippet",
        "input": '''class BatteryConfig(pydantic.BaseModel):
            """Battery asset configuration."""

            name: str
            charge_power_mw: float
            discharge_power_mw: float
            capacity_mwh: float
            efficiency_pct: float
            initial_charge_mwh: float = 0.0
            final_charge_mwh: float | None = 0.0
            freq_mins: int
            ''',
        "output": '''@pydantic.field_validator("name")
            @classmethod
            def check_name(cls, name: str) -> str:
                """Ensure we can identify this asset correctly.

                Args:
                    name: asset name.

                Returns:
                    The asset name.
                """
                assert "battery" in name
                return name
        ''',
    }
]

base = pathlib.Path.home() / "energy-py-linear"

fis = []
for fi in base.rglob("*.py"):
    fis.append(fi)

tr_fis = []
te_fis = []
for fi in fis:
    for pattern in ["evs", "chp"]:
        if pattern in str(fi):
            te_fis.append(fi)
        else:
            tr_fis.append(fi)


def extract_prompts(fis):
    data = []
    for fi in fis:
        code = fi.read_text()
        chunk_size = 200

        for i in range(0, len(code), chunk_size):
            inp = code[i : i + chunk_size]
            out = code[i + chunk_size : i + 2 * chunk_size]

            if len(out) > 12:
                data.append(
                    {
                        "instruction": "Complete the code snippet",
                        "input": inp,
                        "output": out,
                    }
                )
                data.append(
                    {
                        "instruction": "Guess the file name based on the following code snippet.",
                        "input": inp,
                        "output": str(fi.relative_to(pathlib.Path.home())),
                    }
                )

    ds = collections.defaultdict(list)
    for d in data:
        ds["instruction"].append(d["instruction"])
        ds["input"].append(d["input"])
        ds["output"].append(d["output"])
        ds["prompt"].append(
            f"## Instruction: {d['instruction']}\n\n## Input: {d['input']}"
        )
        ds["prompt-response"].append(
            f"{ds['prompt'][-1]}\n\n## Response: {d['output']}"
        )

    return ds


tr_ds = extract_prompts(tr_fis)
te_ds = extract_prompts(te_fis)

"""
other tasks

generating tests
- could just grab part of the battery file, and ask it to predict part of the tests for battery?

TODO
- only including files from certain modules
"""

print(f"{len(tr_ds['prompt'])} training examples.")
print(f"{len(te_ds['prompt'])} test examples.")
train = datasets.Dataset.from_dict(tr_ds)
test = datasets.Dataset.from_dict(te_ds)
ds = datasets.DatasetDict({"train": train, "test": test})
ds.push_to_hub("adgefficiency/energy-py-linear")
