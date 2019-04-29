// TODO: replace panics with proper error handling
use std::error::Error;
use std::fmt::{Debug, Display};
use std::io;
use std::io::prelude::*;
use std::thread;

use atoi::FromRadix10;
use crossbeam_channel::unbounded;
use hashbrown::HashMap;
use itoa;
use num_cpus;

use byteorder::{LittleEndian, ReadBytesExt};
// use bytes::buf::Writer;
use std::sync::Arc;

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

const chrom_idx: usize = 0;
const pos_idx: usize = 1;
const id_idx: usize = 2;
const ref_idx: usize = 3;
const alt_idx: usize = 4;
const qual_idx: usize = 5;
const filter_idx: usize = 6;
const info_idx: usize = 7;
const format_idx: usize = 8;

const NotTrTv: u8 = b'0';
const Tr: u8 = b'1';
const Tv: u8 = b'2';

const SNP: &[u8] = b"SNP";
const INS: &[u8] = b"INS";
const DEL: &[u8] = b"DEL";
const MNP: &[u8] = b"MNP";
const MULTI: &[u8] = b"MULTIALLELIC";
const DSNP: &[u8] = b"DENOVO_SNP";
const DINS: &[u8] = b"DENOVO_INS";
const DDEL: &[u8] = b"DENOVO_DEL";
const DMULTI: &[u8] = b"DENOVO_MULTIALLELIC";

const A: u8 = 'A' as u8;
const C: u8 = 'C' as u8;
const G: u8 = 'G' as u8;
const T: u8 = 'T' as u8;

lazy_static! {
    static ref ACTG: [u8; 256] = {
        let mut actg = [0; 256];

        actg[A as usize] = 1;
        actg[C as usize] = 1;
        actg[G as usize] = 1;
        actg[T as usize] = 1;

        actg
    };
    static ref TSTV: [[u8; 256]; 256] = {
        let mut all: [[u8; 256]; 256] = [[0u8; 256]; 256];
        let mut ref_a = [0; 256];
        let mut ref_c = [0; 256];

        ref_a[G as usize] = Tr;
        ref_a[C as usize] = Tv;
        ref_a[T as usize] = Tv;

        ref_c[T as usize] = Tr;
        ref_c[A as usize] = Tv;
        ref_c[G as usize] = Tv;

        all[A as usize] = ref_a;
        all[G as usize] = ref_a;
        all[C as usize] = ref_c;
        all[G as usize] = ref_c;

        all
    };
}

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

    //https://users.rust-lang.org/t/trim-string-in-place/15809/9
    line.truncate(line.trim_end().len());

    line
}

fn get_tr_tv(refr: u8, alt: u8) -> u8 {
    TSTV[refr as usize][alt as usize]
}

fn snp_is_valid(alt: &[u8]) -> bool {
    return alt[0] == b'A' || alt[0] == b'G' || alt[0] == b'C' || alt[0] == b'T';
}

fn alt_is_valid(alt: &[u8]) -> bool {
    for i in 0..alt.len() {
        if !(alt[i] == b'A' || alt[i] == b'G' || alt[i] == b'C' || alt[i] == b'T') {
            return false;
        }
    }

    return true;
}

fn filter_passes(
    filter_field: &[u8],
    allowed_filters: &HashMap<&[u8], bool>,
    excluded_filters: &HashMap<&[u8], bool>,
) -> bool {
    (allowed_filters.len() == 0 || allowed_filters.contains_key(filter_field))
        && (excluded_filters.len() == 0 || !excluded_filters.contains_key(filter_field))
}

type SNP<'a> = (&'a [u8], &'a [u8], &'a u8, &'a u8);

enum VariantEnum<'a> {
    Snp(SNP<'a>),
    // Del((&'a [u8], u32, &'a u8, Vec<u8>)),
    // Ins((&'a [u8], u32, &'a u8, Vec<u8>)),
    Multi((&'a [u8], Vec<u32>, Vec<&'a u8>, Vec<Vec<u8>>)),
    None,
}

// fn process_snp
// type VariantType<'a> = &'a Variants<'a>;

// site_type, positions, refs, alts, altIndices
fn get_alleles<'a>(pos: &'a [u8], refr: &'a [u8], alt: &'a [u8]) -> VariantEnum<'a> {
    if alt.len() == 1 && refr.len() == 1 {
        if !snp_is_valid(alt) {
            return VariantEnum::None;
        }

        // TODO: Do we need this check
        if alt[0] == refr[0] {
            return VariantEnum::None;
        }

        return VariantEnum::Snp((SNP, pos, &refr[0], &alt[0]));
    }

    let mut positions: Vec<u32> = Vec::new();
    let mut references: Vec<&u8> = Vec::new();
    let mut alleles: Vec<Vec<u8>> = Vec::new();

    let pos = u32::from_radix_10(pos).0;

    if pos == 0 {
        //TODO: error
        return VariantEnum::None;
    }

    let mut n_alleles = 0;
    for t_alt in alt.split(|byt| *byt == b',') {
        n_alleles += 1;

        if !alt_is_valid(alt) {
            continue;
        }

        if refr.len() == 1 {
            // Just a SNP in a multiallelic
            if t_alt.len() == 1 {
                positions.push(pos);
                references.push(&refr[0]);
                alleles.push(Vec::from(t_alt));
                continue;
            }

            // A complex INS
            if t_alt[0] != refr[0] {
                // log.Printf("%s:%s ALT #%d %s", chrom, pos, altIdx+1, insError1)
                continue;
            }

            // A simple INS
            // Pos doesn't change, as pos refers to first ref base
            positions.push(pos);
            references.push(&refr[0]);

            alleles.push(Vec::from(&t_alt[1..t_alt.len()]));

            continue;
        }

        // Simple deletion
        if alt.len() == 1 {
            // complex deletion, but most likely error
            if t_alt[0] != refr[0] {
                // TODO: log
                continue;
            }

            // We use 0 padding for deletions, showing 1st deleted base as ref
            // Therefore need to shift pos & ref by 1
            positions.push(pos + 1);
            references.push(&refr[1]);

            // TODO: error handling/log
            let mut allele = Vec::new();
            itoa::write(&mut allele, 1 - refr.len() as i32).unwrap();
            alleles.push(allele);
        }

        // If we're here, ref and alt are both > 1 base long
        // could be a weird SNP (multiple bases are SNPS, len(ref) == len(alt))
        // could be a weird deletion/insertion
        // could be a completely normal multiallelic (due to padding, shifted)

        //1st check for MNPs and extra-padding SNPs
        if refr.len() == t_alt.len() {
            for i in 0..refr.len() {
                if refr[i] != t_alt[i] {
                    positions.push(pos + i as u32);
                    references.push(&refr[i]);

                    let mut allele = Vec::with_capacity(1);
                    allele.push(t_alt[i]);
                    alleles.push(allele);
                }
            }

            continue;
        }

        // Find the allele representation that minimizes padding, while still checking
        // that the site isn't a mixed type (indel + snp) and checking for intercolation
        // Essentially, Occam's Razor for padding: minimize the number of steps away
        // from left edge to explan the allele
        // EX:
        // If ref == AATCG
        // If alt == AG
        // One interpretation of this site is mixed A->G -3 (-TCG)
        // Another is -3 (-ATC) between the A (0-index) and G (4-index) in ref
        // We prefer the latter approach
        // Ex2: ref: TT alt: TCGATT
        // We prefer +CGAT

        // Like http://www.cureffi.org/2014/04/24/converting-genetic-variants-to-their-minimal-representation/
        // we will use a simple heuristic:
        // 1) For insertions, figure out the shared right edge, from 1 base downstream of first ref base
        // Then, check if the remaining ref bases match the left edge of the alt
        // If they don't, skip that site
        // 2) For deletions, the same, except replace the role of ref with the tAlt

        // Our method should be substantially faster, since we don't need to calculate
        // the min(len(ref), len(tAlt))
        // and because we don't create a new slice for every shared ref/alt at right edges and left

        if t_alt.len() > refr.len() {
            let mut r_idx = 0;

            while t_alt.len() + r_idx > 0
                && refr.len() + r_idx > 1
                && t_alt[t_alt.len() + r_idx - 1] == refr[refr.len() + r_idx - 1]
            {
                r_idx -= 1;
            }

            println!("r+idx {}", r_idx);
            // Then, we require an exact match from left edge, for the difference between the
            // length of the ref, and the shared suffix
            // Ex: alt: TAGCTT ref: TAT
            // We shared 1 base at right edge, so expect that len(ref) - 1, or 3 - 1 = 2 bases of ref
            // match the left edge of alt
            // Here that is TA, for an insertion of +GCT
            // Ex2: alt: TAGCAT ref: TAT
            // Here the AT of the ref matches the last 2 bases of alt
            // So we expect len(ref) - 2 == 1 base of ref to match left edge of the alt (T), for +AGC
            // Ex3: alt TAGTAT ref: TAT
            // Since our loop doesn't check the last base of ref, as in ex2, +AGC
            // This mean we always prefer a 1-base padding, when possible
            // Ex4: alt TAGTAT ref: TGG
            // In this case, we require len(ref) - 0 bases in the ref to match left edge of alt
            // Since they don't (TAG != TGG), we call this complex and move on

            // Insertion
            // If pos is 100 and ref is AATCG
            // and alt is AAAAATCG (len == 7)
            // we expect lIdx to be 2
            // and rIdx to be -3
            // alt[2] is the first non-ref base
            // and alt[len(alt) - 3] == alt[4] is the last non-ref base
            // The position is intPos + lIdx or 100 + 2 - 1 == 101 (100, 101 are padding bases,
            // and we want to keep the last reference base
            // The ref is ref[2 - 1] or ref[1]
            let offset = refr.len() + r_idx;

            if refr[0..offset] != t_alt[0..offset] {
                // TODO: LOG
                continue;
            }

            // position is offset by len(ref) + 1 - rIdx
            // ex1: alt: TAGCTT ref: TAT
            // here we match the first base, so -1
            // we require remainder of left edge to be present,
            // or len(ref) - 1 == 2
            // so intPos + 2 - 1 for last padding base (the A in TA) (intPos + 2 is first unique base)
            positions.push(pos + offset as u32 - 1);
            references.push(&refr[offset - 1]);

            // Similarly, the alt allele starts from len(ref) + rIdx, and ends at len(tAlt) + rIdx
            // from ex: TAGCTT ref: TAT :
            // rIdx == -1 , real alt == tAlt[len(ref) - 1:len(tAlt) - 1] == tALt[2:5]
            // let mut allele = Vec::new();
            // allele.push(t_alt[i]);
            alleles.push(Vec::from(&refr[offset..t_alt.len() + r_idx]));

            continue;
        }

        // Deletion
        // If pos is 100 and alt is AATCG
        // and ref is AAAAATCG (len == 7)
        // we expect lIdx to be 2
        // and rIdx to be -3
        // and alt is -3 or len(ref) + rIdx - lIdx == 8 + -3 - 2
        // position is the first deleted base, or intPos + lIdx == 100 + 2 == 102
        // where (100, 101) are the two padding bases
        // ref is the first deleted base or ref[lIdx] == ref[2]

        // Just like insertion, but try to match all bases from 1 base downstream of tAlt to ref
        let mut r_idx = 0;
        // println!("before r_idx {}", r_idx);
        while t_alt.len() + r_idx > 1
            && refr.len() + r_idx > 0
            && t_alt[t_alt.len() + r_idx - 1] == refr[refr.len() + r_idx - 1]
        {
            r_idx -= 1;
        }

        // println!("after r_idx {}", r_idx);

        let offset = t_alt.len() + r_idx;
        // println!("offset {}", offset);
        if refr[0..offset] != t_alt[0..offset] {
            //TODO: LOG
            continue;
        }
        // println!("past {} {}", refr.len(), offset);
        positions.push(pos + offset as u32);
        // we want the base after the last shared
        references.push(&refr[offset]);

        let mut allele = Vec::new();
        itoa::write(
            &mut allele,
            -(refr.len() as i32 + r_idx as i32 - offset as i32),
        )
        .unwrap();
        alleles.push(allele);
    }

    if n_alleles == 0 || alleles.len() == 0 {
        return VariantEnum::None;
    }

    if n_alleles > 1 {
        return VariantEnum::Multi((MULTI, positions, references, alleles));
    }

    if alleles[0].len() > 1 {
        // println!("before");
        if alleles[0][0] == b'-' {
            // println!("in");
            return VariantEnum::Multi((DEL, positions, references, alleles));
        }

        return VariantEnum::Multi((INS, positions, references, alleles));
    }

    // A 1 allele site where the variant was an MNP
    // They may be sparse or complete, so we empirically check for their presence
    // If the MNP is really just a snp, there is only 1 allele, and reduces to snp
    // > 1, these are labeled differently to allow people to jointly consider the effects
    // of the array of SNPs, since we at the moment consider their effects only independently
    // (which has advantages for CADD, phyloP, phastCons, clinvar, etc reporting)
    if alleles.len() > 1 {
        return VariantEnum::Multi((MNP, positions, references, alleles));
    }

    // TODO: check that this is righ
    return VariantEnum::Multi((SNP, positions, references, alleles));
}

fn process_lines(header: &Vec<String>, rows: Vec<Vec<u8>>) -> usize {
    let n_samples = header.len() - 9;
    let n_fields = header.len();

    let mut multiallelic = false;
    let mut homs: Vec<&str> = Vec::new();
    let mut hets: Vec<&str> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();

    let mut effective_samples: f64;
    let mut ac: u32;
    let mut an: u32;

    let empty_field = "!";
    let field_delim = ";";
    let keep_pos = true;
    let keep_id = false;
    let keep_info = false;

    let mut allowed_filters: HashMap<&[u8], bool> = HashMap::new();
    allowed_filters.insert(b".", true);
    allowed_filters.insert(b"PASS", true);

    let excluded_filters: HashMap<&[u8], bool> = HashMap::new();
    let mut n_count = 0;
    let mut snp_count = 0;
    let mut multi_count = 0;
    for row in rows.iter() {
        // let row: Vec<&str> = row.split("\t").map(|field| field).collect();

        let mut chrom: &[u8] = b"";
        let mut pos: &[u8] = b"";
        let mut refr: &[u8] = b"";
        let mut alt: &[u8] = b"";

        for (idx, field) in row.split(|byt| *byt == b'\t').enumerate() {
            if idx == chrom_idx {
                chrom = field;
                continue;
            }

            if idx == pos_idx {
                pos = field;
                continue;
            }

            if idx == ref_idx {
                refr = field;
                continue;
            }

            if idx == alt_idx {
                alt = field;
                continue;
            }

            if idx == filter_idx {
                // println!("STUFF: {} {} ", refr, alt);

                if !filter_passes(field, &allowed_filters, &excluded_filters) {
                    break;
                }
            }

            if idx > filter_idx {
                // println!("STUFF: {} {} ", refr, alt);

                // let variants = match get_alleles(pos, refr, alt) {
                //     Some(v) => v,
                //     None => break,
                // };

                // if let v = Variants::Many {}

                match get_alleles(pos, refr, alt) {
                    //
                    VariantEnum::Snp(v) => snp_count += 1,
                    VariantEnum::Multi(v) => multi_count += 1,
                    _ => break,
                };

                // println!("{:?}", std::str::from_utf8(variants.1).unwrap());

                // if n_samples == 0 {}

                // match variants {
                //     Variants::One {
                //         site_type,
                //         pos,
                //         refs,
                //         alts,
                //     } => println!("{}", site_type),
                //     Variants::Many {
                //         site_type,
                //         pos,
                //         refs,
                //         alts,
                //     } => println!("NOPE"),
                // }
            }
        }

        // if idx != n_fields {
        //     panic!("Row too short");
        // }

        // let (type, pos, refs, alts, alt_idxs) := get_alleles(record[chrom_idx], record[pos_idx], record[ref_idx], record[alt_idx])

        n_count += 1;
    }

    // println!("{} {} ", snp_count, multi_count);

    n_count
}

fn main() -> Result<(), std::io::Error> {
    let (s1, r1) = unbounded();
    let (s2, r2) = unbounded();
    let n_cpus = num_cpus::get();

    let stdin = io::stdin();
    let mut stdin_lock = stdin.lock();

    let mut lines: Vec<Vec<u8>> = Vec::with_capacity(64);
    let mut len: usize;
    let mut n_count = 0;

    let header: Arc<Vec<String>> = Arc::new(
        get_header(&mut stdin_lock)
            .split("\t")
            .map(|field| field.to_string())
            .collect(),
    );

    println!("HEADER: {:?}", header);

    if header.len() == 9 {
        info!("Found 9 header fields. When genotypes present, we expect 1+ samples after FORMAT (10 fields minimum)")
    }

    for i in 0..n_cpus {
        println!("Spawning thread {}", i);
        let r = r1.clone();
        let s = s2.clone();
        let header = Arc::clone(&header);

        thread::spawn(move || {
            let mut message: Vec<Vec<u8>>;
            let mut n_count: usize = 0;

            loop {
                message = match r.recv() {
                    Ok(v) => v,
                    Err(_) => break,
                };

                n_count += process_lines(&header, message);
            }
            s.send(n_count).unwrap();
        });
    }

    loop {
        let mut buf: Vec<u8> = Vec::new();
        len = stdin_lock.read_until(0xA, &mut buf)?;

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
