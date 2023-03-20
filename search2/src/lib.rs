use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> String {
    return String::from("Hello, world!");
}

#[pymodule]
fn search(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}