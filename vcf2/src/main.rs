use std::io;
use std::io::prelude::*;
use std::thread;

use std::error::Error;
use std::fmt::{Debug, Display};

use crossbeam_channel::unbounded;
use num_cpus;

pub trait FormatError: Debug + Display {
    fn description(&self) -> &str {
        "Wrong header"
    }
}

// TODO: Add error handling
fn get_header<T: BufRead>(reader: &mut T) -> String {
    let mut line: String = String::with_capacity(10_000);
    reader.read_line(&mut line).unwrap();

    if !line.starts_with("##fileformat=VCFv4") {
        panic!("File format not supported: {}", line);
    }

    line.clear();

    loop {
        reader.read_line(&mut line).unwrap();
        if line.starts_with("#CHROM") {
            break;
        }

        if !line.starts_with("#") {
            panic!("Not a VCF file")
        }

        line.clear();
    }

    line
}

fn process_lines(rows: Vec<String>) -> usize {
    let mut n_count = 0;
    for row in rows.iter() {
        // println!("{}", row);
        n_count += 1;
    }

    n_count
}

fn main() -> Result<(), std::io::Error> {
    let (s1, r1) = unbounded();
    let (s2, r2) = unbounded();
    let n_cpus = num_cpus::get();

    for i in 0..n_cpus {
        println!("Spawning thread {}", i);
        let r = r1.clone();
        let s = s2.clone();

        thread::spawn(move || {
            let mut message: Vec<String>;
            let mut n_count: usize = 0;
            loop {
                message = match r.recv() {
                    Ok(v) => v,
                    Err(_) => break,
                };

                n_count += process_lines(message);
            }
            s.send(n_count).unwrap();
        });
    }

    let stdin = io::stdin();
    let mut stdin_lock = stdin.lock();

    let mut lines: Vec<String> = Vec::with_capacity(128);
    let mut len: usize;
    let mut n_count = 0;

    let header = get_header(&mut stdin_lock);

    println!("HEADER: {}", header);
    loop {
        let mut buf: String = String::new();
        len = stdin_lock.read_line(&mut buf)?;

        if len == 0 {
            if lines.len() > 0 {
                s1.send(lines).unwrap();
            }
            break;
        }

        lines.push(buf);
        n_count += 1;

        if lines.len() > 64 {
            s1.send(lines).unwrap();
            lines = Vec::with_capacity(64);
        }
    }

    drop(s1);
    let mut total = 0;
    let mut thread_completed = 0;
    loop {
        thread_completed += 1;
        total += r2.recv().unwrap();

        println! {"Threads {} completed", thread_completed};

        if thread_completed == n_cpus {
            break;
        }
    }

    assert_eq!(total, n_count);

    return Ok(());
}
