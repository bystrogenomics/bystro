use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn annotate(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn annotator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(annotate, m)?)?;
    Ok(())
}

#[pymodule]
fn bystro(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_wrapped(wrap_pymodule!(annotator))?;

    // https://github.com/PyO3/pyo3/issues/2644
    // https://github.com/PyO3/pyo3/tree/main/examples/maturin-starter
    // Inserting to sys.modules allows importing submodules nicely from Python
    let sys = PyModule::import(_py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("bystro.annotator", m.getattr("annotator")?)?;

    Ok(())
}
