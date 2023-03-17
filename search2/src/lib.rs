use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(FromPyObject)]
enum OpensearchMapping  {
    Mapping(HashMap<String, OpensearchMapping>),
    String(String),
    Float(f64),
    Integer(f32)
  }

#[pyfunction]
fn get_boolean_mappings(map_ref: HashMap<String, OpensearchMapping>, parent_name: Option<String>) -> String {
    let mut boolean_mappings: Vec<String> = Vec::new();

    for (property_name, property_value) in map_ref.iter() {
    //     if my_map.contains_key("key1") {

    // for (property_name, property_value) in map_ref.iter() {
    //     if let Some(fields) = property_value.get("fields") {
    //         if let serde_json::Value::Object(sub_props) = fields {
    //             for (sub_prop, sub_prop_value) in sub_props.iter() {
    //                 if let Some(serde_json::Value::String(sub_prop_type)) = sub_prop_value.get("type") {
    //                     if sub_prop_type == "boolean" {
    //                         boolean_mappings.push(format!("{}.{}", parent_name.as_deref().unwrap_or_default(), property_name));
    //                     }
    //                 }
    //             }
    //         }
    //     } else if let Some(properties) = property_value.get("properties") {
    //         boolean_mappings.extend(get_boolean_mappings(properties.as_object().unwrap(), Some(property_name.to_string())));
    //     } else if let Some(serde_json::Value::String(property_type)) = property_value.get("type") {
    //         if property_type == "boolean" {
    //             boolean_mappings.push(format!("{}.{}", parent_name.as_deref().unwrap_or_default(), property_name));
    //         }
    //     }
    }

    // boolean_mappings
    let s = String::from("yes");
    return s
}

// #[pyfunction]
// fn py_get_boolean_mappings(map_ref: &PyDict, parent_name: Option<&str>) -> PyResult<Vec<String>> {
//     let gil = Python::acquire_gil();
//     let py = gil.python();

//     let map_ref = map_ref.as_ref(py)?;
//     let map_ref = serde_json::from_str(map_ref.to_string(py).as_str())?;

//     Ok(get_boolean_mappings(&map_ref, parent_name.map(|s| s.to_string())))
// }

// fn get_boolean_mappings(map_ref: &Value, parent_name: &str) -> Vec<String> {
//     let mut boolean_mappings = Vec::new();
//     if let Some(properties) = map_ref.get("properties") {
//         if let Value::Object(props) = properties {
//             for (property, value) in props {
//                 if let Some(fields) = value.get("fields") {
//                     if let Value::Object(sub_props) = fields {
//                         for (sub_prop, sub_prop_value) in sub_props {
//                             if let (Some(sub_prop_type), Some(property_name)) =
//                                 (sub_prop_value.get("type"), parent_name)
//                             {
//                                 if sub_prop_type == "boolean" {
//                                     boolean_mappings.push(format!("{}.{}", property_name, sub_prop));
//                                 }
//                             }
//                         }
//                     }
//                 } else if let Some(properties) = value.get("properties") {
//                     boolean_mappings.append(&mut get_boolean_mappings(properties, property));
//                 } else if let Some(property_type) = value.get("type") {
//                     if property_type == "boolean" {
//                         boolean_mappings.push(property.to_owned());
//                     }
//                 }
//             }
//         }
//     }
//     boolean_mappings
// }

// #[pyfunction]
// fn rust_get_boolean_mappings(map_ref: &str, parent_name: &str) -> PyResult<Vec<String>> {
//     let map_ref: Value = serde_json::from_str(map_ref)?;
//     let boolean_mappings = get_boolean_mappings(&map_ref, parent_name);
//     Ok(boolean_mappings)
// }

#[pymodule]
fn search(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_boolean_mappings, m)?)?;
    // m.add_function(wrap_pyfunction!(get_boolean_headers, m)?)?;

    Ok(())
}