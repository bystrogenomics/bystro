use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> String {
    return String::from("Hello, world!");
}

#[pymodule]
fn bystro(_py: Python, m: &PyModule) -> PyResult<()> {
    register_child_modules(_py, m)?;
    Ok(())
}

fn register_child_modules(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let search = PyModule::new(py, "search")?;
    let index = PyModule::new(py, "index")?;
    search.add_function(wrap_pyfunction!(hello_world, search)?)?;
    index.add_function(wrap_pyfunction!(hello_world, index)?)?;
    parent_module.add_submodule(search)?;
    parent_module.add_submodule(index)?;
    Ok(())
}
