from pathlib import Path
import pandas as pd

def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--in", dest="in_path", required=True, help="Input PS CSV path")
    # ap.add_argument("--out", dest="out_one_row",
    #                 default="rv_and_transit_planets_evidence.csv",
    #                 help="Output CSV (one row per planet)")
    # ap.add_argument("--out-all", dest="out_all_rows",
    #                 default="rv_and_transit_planets_evidence_all_rows.csv",
    #                 help="Output CSV (all rows for matching planets)")
    # args = ap.parse_args()

    in_path = "exoplanet_data.csv"
    out_one_row = "exoplanet_rv_evidence_with_star_rad.csv"
    out_all_rows = "exoplanet_rv_and_transit_evidence_all_rows.csv"

    # PS CSVs begin with '#' comment lines (metadata)
    df = pd.read_csv(in_path, comment="#", na_values=["NaN", "nan", " "])

    df["st_rad"] = pd.to_numeric(df["st_rad"], errors="coerce")

    # Sanity checks
    required_cols = ["pl_name", "discoverymethod", "pl_bmassprov", "pl_rade", "pl_radj"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # We can still proceed without pl_bmassprov/pl_rade/pl_radj, but warn loudly
        print(f"[WARN] Missing columns: {missing}. Script will proceed with what’s available.")

    # Normalize method text safely
    dm = df.get("discoverymethod", pd.Series([None]*len(df)))
    dm_norm = dm.astype(str).str.strip().str.lower()

    # Transit evidence: method or measured radius
    has_star_rad = (
        df.get("st_rad", pd.Series([None]*len(df))).notna()
    )

    df["_has_star_rad"] = has_star_rad

    print("test", df[["_has_star_rad", "st_rad"]])

    # RV evidence: method or Msini provenance
    bmassprov = df.get("pl_bmassprov", pd.Series([None]*len(df))).astype(str)
    has_rv = (
        dm_norm.eq("radial velocity")
        | bmassprov.str.contains("msini", case=False, na=False)
    )

    # df["_has_transit_row"] = has_transit
    df["_has_rv_row"] = has_rv

    # Aggregate by planet
    by_planet = (
        df.groupby("pl_name", dropna=False)
          .agg(
            #   has_transit=("_has_transit_row", "any"),
              has_star_rad=("_has_star_rad", "any"),
              has_rv=("_has_rv_row", "any"),
              n_rows=("pl_name", "size")
          )
          .reset_index()
    )

    both = by_planet.query("has_rv and has_star_rad")["pl_name"]
    print(both)
    all_rows = df[df["pl_name"].isin(both) & df["st_rad"].notna()].copy()

    # One row per planet (first occurrence)
    one_row = (
        all_rows.sort_values("pl_name")
                .drop_duplicates(subset=["pl_name"], keep="first")
                .copy()
    )

    # Save
    Path(out_one_row).parent.mkdir(parents=True, exist_ok=True)
    Path(out_all_rows).parent.mkdir(parents=True, exist_ok=True)
    one_row.to_csv(out_one_row, index=False)
    all_rows.to_csv(out_all_rows, index=False)

    # Report
    print(f"Total rows read: {len(df)}")
    print(f"Planets with BOTH RV+Transit evidence: {len(one_row)}")
    print(f"Wrote one-row-per-planet to: {out_one_row}")
    print(f"Wrote all-rows to:           {out_all_rows}")
    print(one_row['st_rad'])
    print(one_row['pl_bmassj'])

if __name__ == "__main__":
    main()