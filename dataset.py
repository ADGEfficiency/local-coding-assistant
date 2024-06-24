import datasets

data = [
    {
        "prompt": '''class BatteryConfig(pydantic.BaseModel):
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
        "completion": '''@pydantic.field_validator("name")
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
ds = datasets.Dataset.from_dict(
    {
        "prompt": [i["prompt"] for i in data],
        "completion": [i["completion"] for i in data],
    }
)
ds.save_to_disk("train")
ds = datasets.Dataset.load_from_disk("train")
ds.push_to_hub("adgefficiency/climate-news-db")
