use pyo3::prelude::*;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::{Child, Command, Stdio};

use serde_json;

// TODO: Finish, use this in the python module
// TODO: read from here
// TODO: write from here
#[pyfunction]
fn make_output_string(array_ref: &Vec<Vec<Option<Vec<Vec<Option<i32>>>>>>, delims: &HashMap<String, char>) -> String {
    let empty_field_char = *delims.get("miss").unwrap();
    let mut output_array: Vec<String> = vec![];

    // Expects an array of row arrays, which contain an for each column, or an undefined value
    for row in array_ref {
        let mut row_vec: Vec<String> = vec![];
        for column in row {
            // Some fields may just be missing; we won't store even the
            // alt/pos [[]] structure for those
            if let Some(c) = column {
                let mut column_vec: Vec<String> = vec![];
                // For now, we don't store multiallelics; top level array is placeholder only
                // With breadth 1
                for position_data in &c[0] {
                    if let Some(p) = position_data {
                        let mut position_vec: Vec<String> = vec![];
                        if p.is_empty() {
                            position_vec.push(empty_field_char.to_string());
                        } else if let Some(ref p_data) = p.get(0) {
                            if p_data.is_empty() {
                                position_vec.push(empty_field_char.to_string());
                            } else {
                                for sub_item in p_data {
                                    if let Some(sub) = sub_item {
                                        let sub_str = if sub.is_empty() {
                                            empty_field_char.to_string()
                                        } else {
                                            sub.iter()
                                                .map(|x| x.to_string())
                                                .collect::<Vec<String>>()
                                                .join(&delims["overlap"].to_string())
                                        };
                                        position_vec.push(sub_str);
                                    } else {
                                        position_vec.push(empty_field_char.to_string());
                                    }
                                }
                            }
                        } else {
                            position_vec.push(p.to_string());
                        }
                        let position_str = position_vec.join(&delims["value"].to_string());
                        column_vec.push(position_str);
                    } else {
                        column_vec.push(empty_field_char.to_string());
                    }
                }
                let column_str = column_vec.join(&delims["pos"].to_string());
                row_vec.push(column_str);
            } else {
                row_vec.push(empty_field_char.to_string());
            }
        }
        let row_str = row_vec.join(&delims["fieldSep"].to_string());
        output_array.push(row_str);
    }
    output_array.join("\n")
}


// Example usage
#[test]
fn test_make_output_string() {
    let input_array: Vec<Vec<Option<Vec<Vec<Option<i32>>>>>> = vec![
        vec![Some(vec![vec![Some(1), Some(2)], vec![Some(3), Some(4)]]), None],
        vec![Some(vec![None, Some(vec![5, 6])]), Some(vec![vec![Some(7), Some(8)], vec![Some(9), Some(10)]])],
    ];
    let mut delims: HashMap<String, char> = HashMap::new();
    delims.insert(String::from("miss"), '-');
    delims.insert(String::from("fieldSep"), '\t');
    delims.insert(String::

}

fn make_log_progress2(
    output_str: String,
    num_lines: usize,
    total_lines: &mut usize,
    tx: &mut beanstalkc::Tube,
    tube_name: &str,
    compress: bool,
) -> Result<(), std::io::Error> {
    let mut output_bytes = output_str.as_bytes().to_vec();
    let compressed_output_bytes: Vec<u8>;

    if compress {
        let mut child = Command::new("pigz")
            .arg("-p")
            .arg("2")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn pigz process");

        {
            let mut child_stdin = child.stdin.take().unwrap();
            child_stdin.write_all(&output_bytes)?;
        }

        let mut child_stdout = child.stdout.take().unwrap();
        compressed_output_bytes = std::io::copy(&mut child_stdout, &mut output_bytes)?;

        drop(child_stdout);
        drop(child);
    } else {
        compressed_output_bytes = output_bytes.clone();
    }

    let write_res = tx.put(
        &compressed_output_bytes,
        beanstalkc::PutRequest {
            priority: 0,
            ..Default::default()
        },
    );

    match write_res {
        Ok(_) => {
            *total_lines += num_lines;
            let msg = format!("{{\"total_lines\": {}}}\n", *total_lines);
            let mut writer = BufWriter::new(std::io::stdout());
            writer.write_all(msg.as_bytes())?;
            writer.flush()?;
            Ok(())
        }
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to write to beanstalkd: {}", e),
        )),
    }
}

fn get_header(field_names: &[String]) -> (Vec<String>, Vec<Vec<String>>) {
    let mut parent_names: Vec<String> = Vec::with_capacity(field_names.len());
    let mut children_or_only: Vec<Vec<String>> = Vec::with_capacity(field_names.len());

    for field_name in field_names {
        if field_name.contains('.') {
            let path: Vec<&str> = field_name.split('.').collect();
            parent_names.push(path[0].to_string());

            if path.len() == 2 {
                children_or_only.push(vec![path[1].to_string()]);
            } else {
                children_or_only.push(path[1..].iter().map(|&s| s.to_string()).collect());
            }
        } else {
            parent_names.push(field_name.clone());
            children_or_only.push(vec![field_name.clone()]);
        }
    }

    (parent_names, children_or_only)
}

fn populate_array_path_from_hash(path: &[String], data_for_end_of_path: &HashMap<String, Value>) -> Value {
    if path.is_empty() {
        return serde_json::to_value(data_for_end_of_path).unwrap();
    }

    let mut current_value = data_for_end_of_path;
    for path_element in path {
        current_value = current_value.get(path_element).unwrap().as_object().unwrap();
    }

    serde_json::to_value(current_value).unwrap()
}

fn make_output_string(array_ref: &[Vec<Option<Vec<Vec<Option<String>>>>>], delims: &HashMap<String, String>) -> String {
    let empty_field_char = delims.get("miss").unwrap();
    let mut output_rows = Vec::with_capacity(array_ref.len());

    for row in array_ref {
        let mut output_columns = Vec::with_capacity(row.len());

        for column in row {
            if let Some(column_data) = column {
                let mut position_strings = Vec::with_capacity(column_data[0].len());

                for position_data in &column_data[0] {
                    if let Some(data) = position_data {
                        let data_string = if data.is_empty() {
                            empty_field_char.clone()
                        } else if data.len() == 1 {
                            data[0].clone().unwrap_or_else(|| empty_field_char.clone())
                        } else {
                            data.iter().map(|opt| opt.clone().unwrap_or_else(|| empty_field_char.clone())).collect::<Vec<_>>().join(&delims["overlap"])
                        };

                        position_strings.push(data_string);
                    } else {
                        position_strings.push(empty_field_char.clone());
                    }
                }

                let column_string = position_strings.join(&delims["pos"]);
                output_columns.push(column_string);
            } else {
                output_columns.push(empty_field_char.clone());
            }
        }

        let row_string = output_columns.join(&delims["fieldSep"]);
        output_rows.push(row_string);
    }

    output_rows.join("\n")
}

fn open_compressed_output_file(filename: &str) -> BufWriter<Command> {
    // Create a pipe to pigz compression
    let pigz = Command::new("pigz")
        .arg("-c") // Compress to stdout
        .stdin(Stdio::piped()) // Pipe from stdin
        .stdout(Stdio::piped()) // Pipe to stdout
        .spawn()
        .unwrap();

    // Open a file handle for the compressed output file
    let mut file = File::create(filename).unwrap();
    let mut writer = BufWriter::new(pigz.stdin.unwrap());

    // Spawn a thread to read from pigz stdout and write to the file
    let mut pigz_stdout = pigz.stdout.unwrap();
    std::thread::spawn(move || std::io::copy(&mut pigz_stdout, &mut file).unwrap());

    writer
}

// Example module
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn save(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}