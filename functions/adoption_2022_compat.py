from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

BASE_COLUMNS = [
    "code",
    "HHcount",
    "Popcount",
    "council",
    "sqkm",
    "hect",
    "masterpc",
    "easting",
    "northing",
    "Shape_Leng",
    "Shape_Area",
    "geometry",
]

GPKG_FILES = {
    "social_grade": "MV609 - National Statistics Socio-economic Classification (NS-SeC) (9) by household tenure - People.gpkg",
    "education": "UV501b - Highest level of qualification by age (5).gpkg",
    "car_ownership": "UV405 - Car or van availability.gpkg",
    "housing": "MV407 - Number of cars or vans by accomodation type - households (3) by household size (4).gpkg",
    "household_composition": "MV104 - Household composition - households (10) by dependent children in household (4).gpkg",
    "economic_activity": "UV601 - Economic activity.gpkg",
    "age": "UV102b - Age (20) by sex.gpkg",
    "population": "outputarea2022_usualresidentpopulation.gpkg",
}

SOCIAL_GRADE_PREFIXES = {
    "AB": "AB.Higher.and.intermediate.managerial.administrative.professional",
    "C1": "C1.Supervisory..clerical..junior.managerial.administrative.professional",
    "C2": "C2.Skilled.manual.workers",
    "DE": "DE.Semi.skilled.and.unskilled.manual.workers..on.state.benefit..unemployed..lowest.grade.workers",
}

SOCIAL_GRADE_GROUPS = {
    "AB": [
        "L1: Employers in large establishments - L3: Higher professional occupations",
    ],
    "C1": [
        "L4: Lower professional and higher technical occupations - L6: Higher supervisory occupations",
        "L7: Intermediate occupations",
        "L15: Full-time students",
    ],
    "C2": [
        "L8: Employers in small establishments - L9: Own account workers",
        "L10: Lower supervisory occupations - L11: Lower technical occupations",
    ],
    "DE": [
        "L12: Semi-routine occupations",
        "L13: Routine occupations",
        "L14.1: Never worked - L14.2: Long-term unemployed",
    ],
}

SOCIAL_GRADE_TENURES = {
    "Owned..Owned.outright": [
        "Owned: Owned outright",
    ],
    "Owned..Owned.with.a.mortgage.or.loan.or.shared.ownership": [
        "Owned: Owned with a mortgage or loan",
        "Owned: Shared ownership (part owned and part rented)",
        "Owned: Shared Equity (e.g. LIFT or Help-to-Buy)",
    ],
    "Social.rented": [
        "Social Rented: Council (LA) or Housing Association/ Registered Social Landlord",
    ],
    "Private.rented.or.living.rent.free": [
        "Private rented: Private landlord or letting agency",
        "Private rented: Other",
        "Lives Rent Free",
    ],
}

EDUCATION_LEVELS = {
    "No.qualifications": ["No qualifications"],
    "Level.1": ["Lower school qualifications"],
    "Level.2": ["Upper school qualifications"],
    "Level.3": ["Apprenticeship qualifications"],
    "Level.4.and.above": [
        "Further Education and sub-degree Higher Education qualifications incl. HNC/HNDs",
        "Degree level qualifications or above",
    ],
}

EDUCATION_AGES = ["16 to 24", "25 to 34", "35 to 49", "50 to 64", "65 and over"]

CAR_OWNERSHIP_LEVELS = {
    "No.cars": ["Number of cars or vans in household: No cars or vans"],
    "One.car": ["Number of cars or vans in household: One car or van"],
    "Two.or.more.cars": [
        "Number of cars or vans in household: Two cars or vans",
        "Number of cars or vans in household: Three cars or vans",
        "Number of cars or vans in household: Four or more cars or vans",
    ],
}

HOUSING_TYPES = {
    "Whole.house.or.bungalow.": ["Whole house or bungalow"],
    "Flat..maisonette.or.apartment..or.mobile.temporary.accommodation.": [
        "Flat, maisonette or apartment",
        "Caravan or other mobile or temporary structure",
    ],
}

HOUSING_CAR_BUCKETS = {
    "No.cars.or.vans": ["Number of cars or vans in household: No cars or vans"],
    "One.car.or.van": ["Number of cars or vans in household: One car or van"],
    "Two.or.more.cars.or.vans": [
        "Number of cars or vans in household: Two cars or vans",
        "Number of cars or vans in household: Three cars or vans",
        "Number of cars or vans in household: Four or more cars or vans",
    ],
}

HOUSING_SIZE_BUCKETS = {
    "One.person.aged.17.or.over.in.household": ["One person"],
    "Two.or.more.people.aged.17.or.over.in.household": [
        "Two people",
        "Three people",
        "Four or more people",
    ],
}

AGE_COLUMN_MAP = {
    ("Female", "0 - 4"): "X0.to.4_Females_Age_by_sex",
    ("Female", "5 - 9"): "X5.to.9_Females_Age_by_sex",
    ("Female", "10 - 14"): "X10.to.14_Females_Age_by_sex",
    ("Female", "15"): "X15_Females_Age_by_sex",
    ("Female", "16 - 17"): "X16.to.17_Females_Age_by_sex",
    ("Female", "18 - 19"): "X18.to.19_Females_Age_by_sex",
    ("Female", "20 - 24"): "X20.to.24_Females_Age_by_sex",
    ("Female", "25 - 29"): "X25.to.29_Females_Age_by_sex",
    ("Female", "30 - 34"): "X30.to.34_Females_Age_by_sex",
    ("Female", "35 - 39"): "X35.to.39_Females_Age_by_sex",
    ("Female", "40 - 44"): "X40.to.44_Females_Age_by_sex",
    ("Female", "45 - 49"): "X45.to.49_Females_Age_by_sex",
    ("Female", "50 - 54"): "X50.to.54_Females_Age_by_sex",
    ("Female", "55 - 59"): "X55.to.59_Females_Age_by_sex",
    ("Female", "60 - 64"): "X60.to.64_Females_Age_by_sex",
    ("Female", "65 - 69"): "X65.to.69_Females_Age_by_sex",
    ("Female", "70 - 74"): "X70.to.74_Females_Age_by_sex",
    ("Female", "75 - 79"): "X75.to.79_Females_Age_by_sex",
    ("Female", "80 - 84"): "X80.to.84_Females_Age_by_sex",
    ("Female", "85 and over"): "X85.and.over_Females_Age_by_sex",
    ("Male", "0 - 4"): "X0.to.4_Males_Age_by_sex",
    ("Male", "5 - 9"): "X5.to.9_Males_Age_by_sex",
    ("Male", "10 - 14"): "X10.to.14_Males_Age_by_sex",
    ("Male", "15"): "X15_Males_Age_by_sex",
    ("Male", "16 - 17"): "X16.to.17_Males_Age_by_sex",
    ("Male", "18 - 19"): "X18.to.19_Males_Age_by_sex",
    ("Male", "20 - 24"): "X20.to.24_Males_Age_by_sex",
    ("Male", "25 - 29"): "X25.to.29_Males_Age_by_sex",
    ("Male", "30 - 34"): "X30.to.34_Males_Age_by_sex",
    ("Male", "35 - 39"): "X35.to.39_Males_Age_by_sex",
    ("Male", "40 - 44"): "X40.to.44_Males_Age_by_sex",
    ("Male", "45 - 49"): "X45.to.49_Males_Age_by_sex",
    ("Male", "50 - 54"): "X50.to.54_Males_Age_by_sex",
    ("Male", "55 - 59"): "X55.to.59_Males_Age_by_sex",
    ("Male", "60 - 64"): "X60.to.64_Males_Age_by_sex",
    ("Male", "65 - 69"): "X65.to.69_Males_Age_by_sex",
    ("Male", "70 - 74"): "X70.to.74_Males_Age_by_sex",
    ("Male", "75 - 79"): "X75.to.79_Males_Age_by_sex",
    ("Male", "80 - 84"): "X80.to.84_Males_Age_by_sex",
    ("Male", "85 and over"): "X85.and.over_Males_Age_by_sex",
}

ECONOMIC_ACTIVITY_ALIASES = {
    "Economically.active..Employee..Part.time_Economic_activity": [
        "Economically Active (excluding full-time students) - Employee - Part-time",
    ],
    "Economically.active..Employee..Full.time_Economic_activity": [
        "Economically Active (excluding full-time students) - Employee - Full-time",
    ],
    "Economically.active..Self.employed_Economic_activity": [
        "Economically Active (excluding full-time students) - Self-employed with employees - Part-time",
        "Economically Active (excluding full-time students) - Self-employed with employees - Full-time",
        "Economically Active (excluding full-time students) - Self-employed without employees - Part-time",
        "Economically Active (excluding full-time students) - Self-employed without employees - Full-time",
    ],
    "Economically.active..Unemployed_Economic_activity": [
        "Economically Active (excluding full-time students) - Unemployed - Available for work",
    ],
    "Economically.active..Full.time.student_Economic_activity": [
        "Economically Active full-time students - Total",
    ],
    "Economically.inactive..Retired_Economic_activity": [
        "Economically inactive - Retired",
    ],
    "Economically.inactive..Student_Economic_activity": [
        "Economically inactive - Student",
    ],
    "Economically.inactive..Looking.after.home.or.family_Economic_activity": [
        "Economically inactive - Looking after home/ family",
    ],
    "Economically.inactive..Long.term.sick.or.disabled_Economic_activity": [
        "Economically inactive - Long term sick or disabled",
    ],
    "Economically.inactive..Other_Economic_activity": [
        "Economically inactive - Other",
    ],
}

HOUSEHOLD_COMPOSITION_ALIASES = {
    "One.person.household..Aged.65.and.over_Household_composition": [
        "One person household: Aged 66 and over__No dependent children in household",
    ],
    "One.person.household..Aged.under.65_Household_composition": [
        "One person household: Aged under 66__No dependent children in household",
    ],
    "One.family.only..All.aged.65.and.over_Household_composition": [
        "One family household: All aged 66 and over__No dependent children in household",
    ],
    "One.family.only..Couple.family..No.children_Household_composition": [
        "One family household: Couple family: No children__No dependent children in household",
    ],
    "One.family.only..Couple.family..With.dependent.children_Household_composition": [
        "One family household: Couple family: With dependent children__Dependent children in household: Youngest aged 0 to 4",
        "One family household: Couple family: With dependent children__Dependent children in household: Youngest aged 5 to 11",
        "One family household: Couple family: With dependent children__Dependent children in household: Youngest aged 12 to 18",
    ],
    "One.family.only..Couple.family..All.children.non.dependent_Household_composition": [
        "One family household: Couple family: All children non-dependent__No dependent children in household",
    ],
    "One.family.only..Lone.parent..With.dependent.children_Household_composition": [
        "One family household: Lone parent: With dependent children__Dependent children in household: Youngest aged 0 to 4",
        "One family household: Lone parent: With dependent children__Dependent children in household: Youngest aged 5 to 11",
        "One family household: Lone parent: With dependent children__Dependent children in household: Youngest aged 12 to 18",
    ],
    "One.family.only..Lone.parent..All.children.non.dependent_Household_composition": [
        "One family household: Lone parent: All children non-dependent__No dependent children in household",
    ],
    "Other.households..With.dependent.children_Household_composition": [
        "Other household types: With dependent children__Dependent children in household: Youngest aged 0 to 4",
        "Other household types: With dependent children__Dependent children in household: Youngest aged 5 to 11",
        "Other household types: With dependent children__Dependent children in household: Youngest aged 12 to 18",
    ],
    "Other.households..Other_Household_composition": [
        "Other household types: Other (including all full-time students and all aged 65 and over)__No dependent children in household",
    ],
}


def _coerce_code(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].astype("string").str.strip()


def _exact_code_match(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    left_codes = set(_coerce_code(left))
    right_codes = set(_coerce_code(right))
    if left_codes != right_codes:
        missing = sorted(left_codes - right_codes)
        extra = sorted(right_codes - left_codes)
        parts = [f"{label} code set does not match the population base."]
        if missing:
            parts.append(f"Missing {len(missing)} code(s), e.g. {', '.join(missing[:5])}.")
        if extra:
            parts.append(f"Unexpected {len(extra)} code(s), e.g. {', '.join(extra[:5])}.")
        raise ValueError(" ".join(parts))


def _series_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric.sum(axis=1)


def _ensure_no_total_targets(target_columns: dict[str, list[str]]) -> None:
    invalid = [
        column
        for columns in target_columns.values()
        for column in columns
        if "total" in column.lower()
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(f"Target column list includes forbidden total columns: {preview}")


def build_2022_adoption_inputs(
    datasets: dict[str, gpd.GeoDataFrame],
) -> tuple[gpd.GeoDataFrame, dict[str, list[str]], dict[str, Any]]:
    """Build a 2022 compatibility GeoDataFrame and target-column mapping."""
    required = set(GPKG_FILES)
    missing = required - set(datasets)
    if missing:
        raise KeyError(f"Missing required 2022 datasets: {sorted(missing)}")

    population = datasets["population"].copy()
    population["code"] = _coerce_code(population)

    base = population[BASE_COLUMNS + ["UsualResidentPopulation"]].copy()
    base = base.rename(columns={"code": "geo_code"})
    base["geo_code"] = base["geo_code"].astype("string")
    base["UsualResidentPopulation"] = pd.to_numeric(
        base["UsualResidentPopulation"], errors="coerce"
    ).fillna(0.0)
    base["Area..hectares._Population_density"] = pd.to_numeric(
        base["hect"], errors="coerce"
    ).fillna(0.0)
    base["Density..number.of.persons.per.hectare._Population_density"] = (
        base["UsualResidentPopulation"]
        / base["Area..hectares._Population_density"].replace(0, pd.NA)
    ).fillna(0.0)

    category_notes: dict[str, Any] = {
        "Social Grade": {
            "proxy": "Mapped 2022 NS-SeC groups onto AB/C1/C2/DE proxy bands and aggregated 8 raw tenure leaves into 4 legacy-style tenure groups.",
            "mapping": SOCIAL_GRADE_GROUPS,
        },
        "Education": {
            "proxy": "Mapped 2022 qualification categories to Level 1 / Level 2 / Level 3 / Level 4+ proxies by age band.",
        },
        "Car Ownership": {
            "proxy": "Collapsed UV405 into No cars / One car / Two or more cars.",
        },
        "Housing": {
            "proxy": "Mapped MV407 into house vs flat/mobile, with household-size proxies and 2+ car aggregation.",
        },
        "Household Composition": {
            "proxy": "Mapped only directly defensible MV104 categories. Couple-family outputs remain combined rather than split into married/cohabiting subtypes.",
        },
        "Economic Activity": {
            "proxy": "Collapsed UV601 leaf categories into the legacy scoring buckets.",
        },
        "Age": {
            "proxy": "Mapped UV102b age-sex bins directly, with 10-14 and 85+ carried as combined bins and 65+ helper columns added for interactions.",
        },
        "Population Density": {
            "proxy": "Derived persons-per-hectare from OA2022 usual resident population and OA area.",
        },
    }

    target_columns: dict[str, list[str]] = {}

    social_grade = datasets["social_grade"].copy()
    _exact_code_match(population, social_grade, "social_grade")
    social_grade["code"] = _coerce_code(social_grade)
    social_aliases: dict[str, pd.Series] = {}
    for grade, source_groups in SOCIAL_GRADE_GROUPS.items():
        prefix = SOCIAL_GRADE_PREFIXES[grade]
        for tenure_alias, tenure_sources in SOCIAL_GRADE_TENURES.items():
            source_columns = []
            for group in source_groups:
                for tenure_source in tenure_sources:
                    source_columns.append(f"{group}__{tenure_source}")
            alias = f"{prefix}_{tenure_alias}_Social_grade_2022"
            social_aliases[alias] = _series_sum(social_grade, source_columns)
    social_frame = pd.DataFrame({"geo_code": social_grade["code"], **social_aliases})
    base = base.merge(social_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Social Grade"] = list(social_aliases)

    education = datasets["education"].copy()
    _exact_code_match(population, education, "education")
    education["code"] = _coerce_code(education)
    education_aliases: dict[str, pd.Series] = {}
    for level_alias, source_levels in EDUCATION_LEVELS.items():
        for age_band in EDUCATION_AGES:
            source_columns = [f"{level}__{age_band}" for level in source_levels]
            alias = f"{level_alias}_{age_band.replace(' ', '.').replace('-', '.').replace('+', 'and.over')}_Education_2022"
            education_aliases[alias] = _series_sum(education, source_columns)
    education_frame = pd.DataFrame({"geo_code": education["code"], **education_aliases})
    base = base.merge(education_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Education"] = list(education_aliases)

    car_ownership = datasets["car_ownership"].copy()
    _exact_code_match(population, car_ownership, "car_ownership")
    car_ownership["code"] = _coerce_code(car_ownership)
    car_aliases: dict[str, pd.Series] = {}
    for alias_prefix, source_columns in CAR_OWNERSHIP_LEVELS.items():
        alias = f"{alias_prefix}_Car_or_van_availability_2022"
        car_aliases[alias] = _series_sum(car_ownership, source_columns)
    car_frame = pd.DataFrame({"geo_code": car_ownership["code"], **car_aliases})
    base = base.merge(car_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Car Ownership"] = list(car_aliases)

    housing = datasets["housing"].copy()
    _exact_code_match(population, housing, "housing")
    housing["code"] = _coerce_code(housing)
    housing_aliases: dict[str, pd.Series] = {}
    for housing_alias, housing_sources in HOUSING_TYPES.items():
        for car_alias, car_sources in HOUSING_CAR_BUCKETS.items():
            for size_alias, size_sources in HOUSING_SIZE_BUCKETS.items():
                source_columns = [
                    f"{car_source}__{housing_source}__{size_source}"
                    for car_source in car_sources
                    for housing_source in housing_sources
                    for size_source in size_sources
                ]
                alias = (
                    f"{housing_alias}_Number.of.cars.or.vans.in.household..{car_alias}_"
                    f"{size_alias}_Accommodation_type_2022"
                )
                housing_aliases[alias] = _series_sum(housing, source_columns)
    housing_frame = pd.DataFrame({"geo_code": housing["code"], **housing_aliases})
    base = base.merge(housing_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Housing"] = list(housing_aliases)

    age = datasets["age"].copy()
    _exact_code_match(population, age, "age")
    age["code"] = _coerce_code(age)
    age_aliases: dict[str, pd.Series] = {}
    for source_key, alias in AGE_COLUMN_MAP.items():
        sex, band = source_key
        source_column = f"{sex}__{band}"
        age_aliases[alias] = _series_sum(age, [source_column])
    age_aliases["X65.and.over_Females_Age_by_sex"] = _series_sum(
        age,
        [
            "Female__65 - 69",
            "Female__70 - 74",
            "Female__75 - 79",
            "Female__80 - 84",
            "Female__85 and over",
        ],
    )
    age_aliases["X65.and.over_Males_Age_by_sex"] = _series_sum(
        age,
        [
            "Male__65 - 69",
            "Male__70 - 74",
            "Male__75 - 79",
            "Male__80 - 84",
            "Male__85 and over",
        ],
    )
    age_frame = pd.DataFrame({"geo_code": age["code"], **age_aliases})
    base = base.merge(age_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Age"] = [alias for alias in age_aliases if "X65.and.over" not in alias]

    economic_activity = datasets["economic_activity"].copy()
    _exact_code_match(population, economic_activity, "economic_activity")
    economic_activity["code"] = _coerce_code(economic_activity)
    activity_aliases = {
        alias: _series_sum(economic_activity, source_columns)
        for alias, source_columns in ECONOMIC_ACTIVITY_ALIASES.items()
    }
    activity_frame = pd.DataFrame({"geo_code": economic_activity["code"], **activity_aliases})
    base = base.merge(activity_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Economic Activity"] = list(activity_aliases)

    household_composition = datasets["household_composition"].copy()
    _exact_code_match(population, household_composition, "household_composition")
    household_composition["code"] = _coerce_code(household_composition)
    household_aliases = {
        alias: _series_sum(household_composition, source_columns)
        for alias, source_columns in HOUSEHOLD_COMPOSITION_ALIASES.items()
    }
    household_frame = pd.DataFrame(
        {"geo_code": household_composition["code"], **household_aliases}
    )
    base = base.merge(household_frame, on="geo_code", how="left", validate="one_to_one")
    target_columns["Household Composition"] = list(household_aliases)

    target_columns["Population Density"] = [
        "Density..number.of.persons.per.hectare._Population_density",
    ]

    _ensure_no_total_targets(target_columns)

    missing_target_columns = [
        column
        for columns in target_columns.values()
        for column in columns
        if column not in base.columns
    ]
    if missing_target_columns:
        preview = ", ".join(missing_target_columns[:5])
        raise ValueError(f"Generated target columns missing from merged dataset: {preview}")

    merged = gpd.GeoDataFrame(base, geometry="geometry", crs=population.crs)

    metadata = {
        "target_counts": {category: len(columns) for category, columns in target_columns.items()},
        "notes": category_notes,
        "row_count": len(merged),
        "column_count": len(merged.columns),
    }
    return merged, target_columns, metadata
