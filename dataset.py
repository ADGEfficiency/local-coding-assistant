import collections
import datasets

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

ds = collections.defaultdict(list)
for d in data:
    ds["instruction"].append(d["instruction"])
    ds["input"].append(d["input"])
    ds["output"].append(d["output"])
    ds["prompt"].append(
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {d['instruction']}\n\n### Input: {d['input']}"
    )
    ds["prompt-response"].append(f"{ds['prompt'][-1]}\n\n### Response: {d['output']}")

train = datasets.Dataset.from_dict(ds)
ds = datasets.DatasetDict({"train": train})
ds.push_to_hub("adgefficiency/energy-py-linear")
