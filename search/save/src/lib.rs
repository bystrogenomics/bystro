use pyo3::prelude::*;
use std::collections::HashMap;
use rustbeanstalkd::{Beanstalkd, BeanstalkdError};
use heed;
fs::create_dir_all("target/heed.mdb")?;
let env = EnvOpenOptions::new().open("target/heed.mdb")?;

// We open the default unamed database.
// Specifying the type of the newly created database.
// Here we specify that the key is an str and the value a simple integer.
let db: Database<Str, OwnedType<i32>> = env.create_database(None)?;

// We then open a write transaction and start writing into the database.
// All of those puts are type checked at compile time,
// therefore you cannot write an integer instead of a string.
let mut wtxn = env.write_txn()?;
db.put(&mut wtxn, "seven", &7)?;
db.put(&mut wtxn, "zero", &0)?;
db.put(&mut wtxn, "five", &5)?;
db.put(&mut wtxn, "three", &3)?;
wtxn.commit()?;

// We open a read transaction to check if those values are available.
// When we read we also type check at compile time.
let rtxn = env.read_txn()?;

let ret = db.get(&rtxn, "zero")?;
assert_eq!(ret, Some(0));

let ret = db.get(&rtxn, "five")?;
assert_eq!(ret, Some(5));
#[pyfunction]
fn read_metadata() -> PyResult<String> {
    Ok("Yeah")
}
// use pyo3::types::{PyDict, PyList};

// use arrow::datatypes::{Schema, Field, DataType};
// use arrow::array::{ArrayRef, StringArray, BooleanArray, Int64Array, Float64Array};
// use arrow::record_batch::RecordBatch;
// use arrow::table::Table;


use serde_json;
#[pyfunction]
use serde_json;

fn deserialize_arguments(input_query_body: &str, index_name: &str, assembly: &str, field_names: &str, index_config: &str, connection: &str) -> (serde_json::Value, String, String, Vec<String>, serde_json::Value, serde_json::Value) {
    /*
    This function takes the required arguments and deserializes them to the specified types.

    Parameters:
    - input_query_body: A string representing the input query body in JSON format.
    - index_name: A string representing the index name.
    - assembly: A string representing the assembly.
    - field_names: A string representing an array of field names in JSON format.
    - index_config: A string representing the index configuration in JSON format.
    - connection: A string representing the connection configuration in JSON format.

    Returns:
    A tuple of deserialized arguments in the order they were passed.
    */
    let input_query_body_deserialized: serde_json::Value = serde_json::from_str(input_query_body).unwrap();
    let field_names_deserialized: Vec<String> = serde_json::from_str(field_names).unwrap();
    let index_config_deserialized: serde_json::Value = serde_json::from_str(index_config).unwrap();
    let connection_deserialized: serde_json::Value = serde_json::from_str(connection).unwrap();

    (input_query_body_deserialized, index_name.to_string(), assembly.to_string(), field_names_deserialized, index_config_deserialized, connection_deserialized)
}

#[test]
fn test_deserialize_arguments() {
    let input_query_body = r#"{"query": {"match_all": {}}}"#;
    let index_name = "my_index";
    let assembly = "my_assembly";
    let field_names = r#"["field1", "field2", "field3"]"#;
    let index_config = r#"{"settings": {"number_of_shards": 1, "number_of_replicas": 0}}"#;
    let connection = r#"{"host": "localhost", "port": 9200}"#;

    let (input_query_body_deserialized, index_name_deserialized, assembly_deserialized, field_names_deserialized, index_config_deserialized, connection_deserialized) = deserialize_arguments(input_query_body, index_name, assembly, field_names, index_config, connection);

    assert_eq!(input_query_body_deserialized, serde_json::json!({"query": {"match_all": {}}}));
    assert_eq!(index_name_deserialized, "my_index");
    assert_eq!(assembly_deserialized, "my_assembly");
    assert_eq!(field_names_deserialized, vec!["field1".to_string(), "field2".to_string(), "field3".to_string()]);
    assert_eq!(index_config_deserialized, serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}}));
    assert_eq!(connection_deserialized, serde_json::json!({"host": "localhost", "port": 9200}));
}

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

use std::collections::HashMap;

use beanstalkc::{Beanstalkc, BeanstalkcError};
use serde_json::{json, Value};
use std::collections::HashMap;

fn make_log_progress(
    output_string: &str,
    num_newlines: usize,
    total_lines: &mut usize,
    bs: &mut Beanstalkc,
    tube: &str,
) -> Result<(), BeanstalkcError> {
    // Increment total lines
    *total_lines += num_newlines;

    // Send JSON message to Beanstalkd with the total number of lines
    let message = json!({
        "total_lines": *total_lines,
    });
    let payload = serde_json::to_vec(&message).unwrap();
    bs.put(&payload, 1, 0, 60, tube)?;

    // Write output string to stdout
    print!("{}", output_string);
    Ok(())
}

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::process::{Child, Command, Stdio};

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

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::{Command, Stdio};

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


fn main() {
    let mut total_lines = 0;

    // Open the compressed output file
    let mut handle = open_compressed_output_file("output.txt.gz");

    // Process the data and write to the compressed output file
    let data = vec![vec![Some(vec![Some(vec![Some(1)])])]];
    let delims = [("miss".to_string(), '-'), ("pos".to_string(), ','), ("overlap".to_string(), ';'), ("value".to_string(), '|'), ("fieldSep".to_string(), '\t')].iter().cloned().collect();
    let output_str = make_output_string(&data, &delims);
    make_log_progress(&output_str, 1, &mut total_lines, &mut handle);
}

// /// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
// #[pyfunction]
// fn json_to_arrow(data: &PyList) -> PyResult<String> {
//     // Loop through the list of dicts
//     for entry in data.iter() {
//         let record = entry.downcast::<PyDict>()?;

//         // Create vectors to hold the data for each field
//         let mut string_data: Vec<Option<String>> = Vec::new();
//         let mut boolean_data: Vec<Option<bool>> = Vec::new();
//         let mut integer_data: Vec<Option<i64>> = Vec::new();
//         let mut float_data: Vec<Option<f64>> = Vec::new();

//         // Loop through the keys in the dictionary
//         for (key, value) in record.items() {
//             // Check the type of the value and add it to the appropriate data vector
//             // if let Ok(string_value) = value.extract::<String>() {
//             //     string_data.push(Some(string_value));
//             // } else if let Ok(boolean_value) = value.extract::<bool>() {
//             //     boolean_data.push(Some(boolean_value));
//             // } else if let Ok(integer_value) = value.extract::<i64>() {
//             //     integer_data.push(Some(integer_value));
//             // } else if let Ok(float_value) = value.extract::<f64>() {
//             //     float_data.push(Some(float_value));
//             // }
//         }
//     }

//     Ok(format!("Got string of length {}", a.len()))
// }

// #[pyfunction]
// fn list_of_dicts_to_arrow_table(py: Python, data: &PyList) -> PyResult<PyObject> {
//     // Get the length of the list
//     let len = data.len();

//     // Create vectors to hold the schema and record batches
//     let mut schema_fields: Vec<Field> = Vec::new();
//     let mut record_batches: Vec<RecordBatch> = Vec::new();

//     // Loop through the list of dicts
//     for i in 0..len {
//         // Get the dictionary at index i
//         let dict = data.get_item(i).downcast::<PyDict>()?;

//         // Create vectors to hold the data for each field
//         let mut string_data: Vec<Option<String>> = Vec::new();
//         let mut boolean_data: Vec<Option<bool>> = Vec::new();
//         let mut integer_data: Vec<Option<i64>> = Vec::new();
//         let mut float_data: Vec<Option<f64>> = Vec::new();

//         // Loop through the keys in the dictionary
//         for (key, value) in dict.items() {
//             // Check the type of the value and add it to the appropriate data vector
//             if let Ok(string_value) = value.extract::<String>() {
//                 string_data.push(Some(string_value));
//             } else if let Ok(boolean_value) = value.extract::<bool>() {
//                 boolean_data.push(Some(boolean_value));
//             } else if let Ok(integer_value) = value.extract::<i64>() {
//                 integer_data.push(Some(integer_value));
//             } else if let Ok(float_value) = value.extract::<f64>() {
//                 float_data.push(Some(float_value));
//             }
//         }

//         // Create fields for each data vector and add them to the schema
//         if !string_data.is_empty() {
//             schema_fields.push(Field::new(key.to_string(), DataType::Utf8, false));
//         } else if !boolean_data.is_empty() {
//             schema_fields.push(Field::new(key.to_string(), DataType::Boolean, false));
//         } else if !integer_data.is_empty() {
//             schema_fields.push(Field::new(key.to_string(), DataType::Int64, false));
//         } else if !float_data.is_empty() {
//             schema_fields.push(Field::new(key.to_string(), DataType::Float64, false));
//         }

//         // Create arrays for each data vector and create a record batch
//         let string_array: ArrayRef = string_data.into();
//         let boolean_array: ArrayRef = boolean_data.into();
//         let integer_array: ArrayRef = integer_data.into();
//         let float_array: ArrayRef = float_data.into();

//         let record_batch = RecordBatch::try_new(
//             Arc::new(Schema::new(schema_fields.clone())),
//             vec![string_array, boolean_array, integer_array, float_array],
//         )?;

//         // Add the record batch to the vector
//         record_batches.push(record_batch);
//     }

//     // Create the Arrow Table from the schema and record batches
//     let arrow_table = Table::try_new(Arc::new(Schema::new(schema_fields)), record_batches)?;

//     // Convert the Arrow Table to a PyArrow Table and return it
//     let pyarrow = py.import("pyarrow")?;
//     let table_class = pyarrow;

/// A Python module implemented in Rust.
#[pymodule]
fn save(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_function(wrap_pyfunction!(json_to_arrow, m)?)?;
    // m.add_function(wrap_pyfunction!(json_to_arrow, m)?)?;
    Ok(())
}