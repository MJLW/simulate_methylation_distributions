use core::panic;
use std::{env, fs::File};

use hypors::{common::TailType, mann_whitney::u_test};
use itertools::Itertools;
use polars::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq};

const SEED: u64 = 0;
const ALPHA: f64 = 0.05;

fn read_pheno(pheno_path: &str) -> LazyFrame {
    let pathogenicity_values = Series::from_iter(vec![
        "Pathogenic".to_string(),
        "Likely pathogenic".to_string(),
    ]);

    return LazyCsvReader::new(pheno_path)
        .with_has_header(true)
        .with_infer_schema_length(Some(10000))
        .with_separator(b'\t')
        .finish()
        .expect("Couldn't parse pheno file.")
        .filter(
            col("Pathogenicity")
                .is_in(lit(pathogenicity_values.clone()))
                .and(col("QC_Flag").eq(false))
                .and(col("Tissue").str().contains(lit("^Blood"), true)),
        )
        .select([col("Person_Stable_ID"), col("Array_ID"), col("Status")]);
}

fn read_betas(beta_path: &str) -> LazyFrame {
    return LazyCsvReader::new(beta_path)
        .with_infer_schema_length(Some(10000))
        .with_has_header(true)
        .with_separator(b'\t')
        .finish()
        .expect("Could not read file as tsv");
}

fn convert_betas_to_vec(df_betas: &DataFrame) -> Vec<Vec<f64>> {
    return (0..df_betas.height())
        .into_iter()
        .map(|i| {
            df_betas
                .get_row(i)
                .unwrap()
                .0
                .iter()
                .map(|val| val.try_extract::<f64>().unwrap())
                .collect_vec()
        })
        .collect_vec();
}

fn get_samples_columns(df_pheno_subset: &DataFrame) -> Vec<Expr> {
    let unique_samples = df_pheno_subset["Array_ID"].clone();
    let target_columns = unique_samples
        .str()
        .unwrap()
        .iter()
        .map(|col_name| col(col_name.unwrap()))
        .collect_vec();

    return target_columns;
}

fn select_indices_as_series(indices: &Vec<usize>, vector: &Vec<f64>, series_name: &str) -> Series {
    return Series::from_vec(
        series_name.into(),
        indices.iter().map(|&i| vector[i]).collect_vec(),
    );
}

fn run_u_test(
    values: &Vec<Vec<f64>>,
    case_indices: &Vec<usize>,
    control_indices: &Vec<usize>,
) -> Vec<f64> {
    return (0..values.len())
        .into_iter()
        .map(|i| {
            u_test(
                &select_indices_as_series(case_indices, &(*values)[i], "cases"),
                &select_indices_as_series(control_indices, &(*values)[i], "controls"),
                ALPHA,
                TailType::Two,
            )
            .unwrap()
            .p_value as f64
        })
        .collect_vec();
}

fn run_h0_combinations(
    df_betas: &DataFrame,
    df_controls: &DataFrame,
    n_iterations: usize,
    n_max_cases: usize,
    n_max_controls: usize,
) -> Vec<Series> {
    let betas = convert_betas_to_vec(&df_betas);

    /* Test H0 hypotheses on controls */
    let mut rng = StdRng::seed_from_u64(SEED);
    let n_samples = df_controls.shape().0;

    let mut combination_series: Vec<Series> = Vec::new();
    for i in 0..n_iterations {
        let case_sampling_size: usize = if n_max_cases < n_samples {
            n_max_cases
        } else {
            panic!(
                "Could not draw {} number of cases, there are only {}",
                n_max_cases, n_samples
            );
        };
        let case_selections: Vec<usize> =
            seq::index::sample(&mut rng, n_samples, case_sampling_size).into_vec();

        let control_sampling_size: usize = if n_max_controls < (n_samples - n_max_cases) {
            n_max_controls
        } else {
            panic!(
                "Could not draw {} number of cases, there are only {}",
                n_max_controls,
                n_samples - n_max_cases
            );
        };
        let control_sampling: Vec<usize> =
            seq::index::sample(&mut rng, control_sampling_size, n_max_controls).into_vec();

        let control_selections: Vec<usize> = (0..n_samples)
            .into_iter()
            .filter(|i| case_selections.contains(i) == false)
            .enumerate()
            .filter(|(i, _)| control_sampling.contains(i))
            .map(|(_, idx)| idx)
            .collect_vec();

        let results: Vec<f64> = run_u_test(&betas, &case_selections, &control_selections);

        combination_series.push(Series::from_vec(format!("iter_{}", i).into(), results));
    }

    return combination_series;
}

fn run_h1_combinations(
    df_betas: &DataFrame,
    df_cases: &DataFrame,
    df_controls: &DataFrame,
    n_iterations: usize,
    n_max_cases: usize,
    n_max_controls: usize,
) -> Vec<Series> {
    let betas = convert_betas_to_vec(&df_betas);

    /* Test H1 hypotheses on cases */
    let mut rng = StdRng::seed_from_u64(SEED);
    let n_cases = df_cases.shape().0;
    let n_controls = df_controls.shape().0;

    let mut combination_series: Vec<Series> = Vec::new();
    for i in 0..n_iterations {
        let case_sampling_size: usize = if n_max_cases < n_cases {
            n_max_cases
        } else {
            panic!(
                "Could not draw {} cases, there are only {}.",
                n_max_cases, n_cases
            );
        };
        let case_selections: Vec<usize> =
            seq::index::sample(&mut rng, n_cases, case_sampling_size).into_vec();

        let control_sampling_size: usize = if n_max_controls < n_controls {
            n_max_controls
        } else {
            panic!(
                "Could not draw {} cases, there are only {}.",
                n_max_controls, n_controls
            );
        };
        let control_selections: Vec<usize> =
            seq::index::sample(&mut rng, n_controls, control_sampling_size).into_vec();

        let results: Vec<f64> = run_u_test(&betas, &case_selections, &control_selections);

        combination_series.push(Series::from_vec(format!("iter_{}", i).into(), results));
    }

    return combination_series;
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mode = &args[1];
    let n_combinations: usize = args[2].parse::<usize>().unwrap();
    let n_target_cases: usize = args[3].parse::<usize>().unwrap();
    let n_target_controls: usize = args[4].parse::<usize>().unwrap();
    let pheno_path = &args[5];
    let betas_path = &args[6];
    let output_path = &args[7];

    let split_mode: Vec<&str> = mode.split(".").collect();
    let test = split_mode[0];

    let lf_pheno = read_pheno(pheno_path);

    if test == "H0" {
        /* H0 */
        /* Get controls */
        let lf_controls = lf_pheno
            .clone()
            .filter(col("Status").str().contains(lit("Negative Control"), true))
            .select([all().exclude(["^Status"])]);

        let df_controls = lf_controls
            .with_row_index("Index", Some(0))
            .collect()
            .unwrap();

        let target_columns = get_samples_columns(&df_controls);

        let df_betas = read_betas(betas_path)
            .select(&target_columns)
            .collect()
            .expect("Failed to select the specified columns");

        let combination_series = run_h0_combinations(
            &df_betas,
            &df_controls,
            n_combinations,
            n_target_cases,
            n_target_controls,
        );

        let columns = combination_series.into_iter().map(Column::from).collect();
        let mut df_output = DataFrame::new(columns).unwrap();

        println!("{}", df_output);

        let mut output_file = File::create(output_path).expect("Could not create output file");
        let _ = CsvWriter::new(&mut output_file)
            .include_header(true)
            .with_separator(b'\t')
            .finish(&mut df_output);
    } else if test == "H1" {
        /* H1 */
        // let gene = "ARID1B";
        let gene = split_mode[1];

        let lf_controls = lf_pheno
            .clone()
            .filter(col("Status").str().contains(lit("Negative Control"), true))
            .select([all().exclude(["^Status"])]);

        let df_controls = lf_controls
            .with_row_index("Index", Some(0))
            .collect()
            .unwrap();

        let lf_cases = lf_pheno
            .clone()
            .filter(col("Status").str().contains(lit(gene), true))
            .select([all().exclude(["^Status"])]);

        let df_cases = lf_cases.with_row_index("Index", Some(0)).collect().unwrap();

        let control_columns = get_samples_columns(&df_controls);
        let case_columns = get_samples_columns(&df_cases);

        let target_columns: Vec<Expr> = control_columns
            .iter()
            .chain(case_columns.iter())
            .cloned()
            .collect();

        let df_betas = read_betas(betas_path)
            .select(&target_columns)
            .collect()
            .expect("Failed to select the specified columns");

        let combination_series = run_h1_combinations(
            &df_betas,
            &df_cases,
            &df_controls,
            n_combinations,
            n_target_cases,
            n_target_controls,
        );
        let columns = combination_series.into_iter().map(Column::from).collect();
        let mut df_output = DataFrame::new(columns).unwrap();

        println!("{}", df_output);

        let mut output_file = File::create(output_path).expect("Could not create output file");
        let _ = CsvWriter::new(&mut output_file)
            .include_header(true)
            .with_separator(b'\t')
            .finish(&mut df_output);
    }
}
