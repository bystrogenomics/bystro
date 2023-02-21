use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
// use arrow::datatypes::{Schema, Field, DataType};
// use arrow::array::{ArrayRef, StringArray, BooleanArray, Int64Array, Float64Array};
// use arrow::record_batch::RecordBatch;
// use arrow::table::Table;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn json_to_arrow(data: &PyList) -> PyResult<String> {
    // Loop through the list of dicts
    for entry in data.iter() {
        let record = entry.downcast::<PyDict>()?;

        // Create vectors to hold the data for each field
        let mut string_data: Vec<Option<String>> = Vec::new();
        let mut boolean_data: Vec<Option<bool>> = Vec::new();
        let mut integer_data: Vec<Option<i64>> = Vec::new();
        let mut float_data: Vec<Option<f64>> = Vec::new();

        // Loop through the keys in the dictionary
        for (key, value) in record.items() {
            // Check the type of the value and add it to the appropriate data vector
            // if let Ok(string_value) = value.extract::<String>() {
            //     string_data.push(Some(string_value));
            // } else if let Ok(boolean_value) = value.extract::<bool>() {
            //     boolean_data.push(Some(boolean_value));
            // } else if let Ok(integer_value) = value.extract::<i64>() {
            //     integer_data.push(Some(integer_value));
            // } else if let Ok(float_value) = value.extract::<f64>() {
            //     float_data.push(Some(float_value));
            // }
        }
    }

    Ok(format!("Got string of length {}", a.len()))
}

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
    m.add_function(wrap_pyfunction!(json_to_arrow, m)?)?;
    // m.add_function(wrap_pyfunction!(json_to_arrow, m)?)?;
    Ok(())
}