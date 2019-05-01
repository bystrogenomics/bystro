// TODO: replace panics with proper error handling
use std::io;
use std::io::prelude::*;
use std::thread;

use atoi::FromRadix10;
use crossbeam_channel::unbounded;
use hashbrown::HashMap;
use itoa;
use memchr::memchr;
use num_cpus;

use std::sync::Arc;

#[macro_use]
extern crate log;

const CHROM_IDX: usize = 0;
const POS_IDX: usize = 1;
// const ID_IDX: usize = 2;
const REF_IDX: usize = 3;
const ALT_IDX: usize = 4;
// const QUAL_IDX: usize = 5;
const FILTER_IDX: usize = 6;
// const INFO_IDX: usize = 7;
const FORMAT_IDX: usize = 8;

const NOT_TSTV: u8 = b'0';
const TS: u8 = b'1';
const TV: u8 = b'2';

const SNP: &[u8] = b"SNP";
const INS: &[u8] = b"INS";
const DEL: &[u8] = b"DEL";
const MNP: &[u8] = b"MNP";
const MULTI: &[u8] = b"MULTIALLELIC";

const A: u8 = b'A';
const C: u8 = b'C';
const G: u8 = b'G';
const T: u8 = b'T';

static TSTV: [[u8; 256]; 256] = {
    let mut all: [[u8; 256]; 256] = [[0u8; 256]; 256];

    all[G as usize][A as usize] = TS;
    all[A as usize][G as usize] = TS;
    all[C as usize][T as usize] = TS;
    all[T as usize][C as usize] = TS;

    all[C as usize][A as usize] = TV;
    all[T as usize][A as usize] = TV;
    all[C as usize][G as usize] = TV;
    all[T as usize][G as usize] = TV;

    all[A as usize][T as usize] = TV;
    all[G as usize][T as usize] = TV;
    all[A as usize][C as usize] = TV;
    all[G as usize][C as usize] = TV;

    all
};

// #[inline]
// fn tstv(u8) {

// }

// TODO: Add error handling
fn get_header<T: BufRead>(reader: &mut T) -> Vec<u8> {
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

    Vec::from(line.as_bytes())
}

fn snp_is_valid(alt: u8) -> bool {
    return alt == b'A' || alt == b'G' || alt == b'C' || alt == b'T';
}

fn alt_is_valid(alt: &[u8]) -> bool {
    if !snp_is_valid(alt[0]) {
        return false;
    }

    for i in 1..alt.len() {
        if !(alt[i] == b'A' || alt[i] == b'G' || alt[i] == b'C' || alt[i] == b'T') {
            return false;
        }
    }

    true
}

fn filter_passes(
    filter_field: &[u8],
    allowed_filters: &HashMap<&[u8], bool>,
    excluded_filters: &HashMap<&[u8], bool>,
) -> bool {
    (allowed_filters.len() == 0 || allowed_filters.contains_key(filter_field))
        && (excluded_filters.len() == 0 || !excluded_filters.contains_key(filter_field))
}

type SnpType<'a> = (&'a [u8], u8, u8);
type DelType<'a> = (u32, u8, i32);
type InsType<'a> = (&'a [u8], u8, &'a [u8]);
type Multi<'a> = (&'a [u8], Vec<u32>, Vec<&'a u8>, Vec<Vec<u8>>);

enum VariantEnum<'a> {
    Snp(SnpType<'a>),
    Del(DelType<'a>),
    Ins(InsType<'a>),
    Multi(Multi<'a>),
    None,
}

fn get_alleles<'a>(pos: &'a [u8], refr: &'a [u8], alt: &'a [u8]) -> VariantEnum<'a> {
    if alt.len() == 1 {
        if !snp_is_valid(alt[0]) {
            return VariantEnum::None;
        }

        if refr.len() == 1 {
            return VariantEnum::Snp((pos, refr[0], alt[0]));
        }

        // simple deletion must have 1 base padding match
        if alt[0] != refr[0] {
            // TODO: Error
            return VariantEnum::None;
        }

        // pos is the next base over (first deleted base)
        // ref is also the first deleted base, since alt is of 1 padding, that's idx 1 (2nd ref base)
        // alt == len(alt) - len(ref) for len(alt) < len(ref)
        // example: alt = A (len == 1), ref = AAATCC (len == 6)
        // 1 - 6 = -5 (then conver to string)
        let pos = u32::from_radix_10(pos).0 + 1;

        return VariantEnum::Del((pos, refr[0], 1 - refr.len() as i32));
    } else if refr.len() == 1 && memchr(b',', alt) == None {
        if !alt_is_valid(alt) {
            return VariantEnum::None;
        }

        if alt[0] == refr[0] {
            return VariantEnum::None;
        }

        return VariantEnum::Ins((pos, refr[0], &alt[1..alt.len()]));
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
    let refr_len = refr.len() as i32;
    let mut r_idx: i32;
    let mut talt_len: i32;

    for t_alt in alt.split(|byt| *byt == b',') {
        n_alleles += 1;

        if !alt_is_valid(t_alt) {
            continue;
        }

        if refr.len() == 1 {
            // Just a SNP in a multiallelic
            if t_alt.len() == 1 {
                positions.push(pos);
                references.push(&refr[0]);
                alleles.push(t_alt.to_vec());
                continue;
            }

            // A complex INS
            if t_alt[0] != refr[0] {
                continue;
            }

            // A simple INS
            // Pos doesn't change, as pos refers to first ref base
            positions.push(pos);
            references.push(&refr[0]);

            alleles.push(t_alt[1..t_alt.len()].to_vec());

            continue;
        }

        //Simple deletion
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

                    alleles.push(vec![t_alt[i]; 1]);
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

        r_idx = 0;
        talt_len = t_alt.len() as i32;
        let offset: usize;
        if t_alt.len() > refr.len() {
            while talt_len + r_idx > 0
                && refr_len + r_idx > 1
                && t_alt[{ talt_len + r_idx - 1 } as usize]
                    == refr[{ refr_len + r_idx - 1 } as usize]
            {
                r_idx -= 1;
            }

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
            offset = { refr_len + r_idx } as usize;

            // TODO: DO we need this check
            // if refr[0..offset] != t_alt[0..offset] {
            //     println!("WTF");
            //     continue;
            // }

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
            alleles.push(t_alt[offset..{ talt_len + r_idx } as usize].to_vec());

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
        // let mut r_idx = 0;
        while talt_len + r_idx > 1
            && refr_len + r_idx > 0
            && t_alt[{ talt_len + r_idx - 1 } as usize] == refr[{ refr_len + r_idx - 1 } as usize]
        {
            r_idx -= 1;
        }

        offset = { talt_len + r_idx } as usize;

        positions.push(pos + offset as u32);
        // we want the base after the last shared
        references.push(&refr[offset]);

        let mut allele = Vec::new();
        itoa::write(&mut allele, -(refr_len + r_idx - offset as i32)).unwrap();
        alleles.push(allele);
    }

    if n_alleles == 0 || alleles.len() == 0 {
        return VariantEnum::None;
    }

    if n_alleles > 1 {
        return VariantEnum::Multi((MULTI, positions, references, alleles));
    }

    // A single allele

    // A 1 allele site where the variant was an MNP
    // They may be sparse or complete, so we empirically check for their presence
    // If the MNP is really just a snp, there is only 1 allele, and reduces to snp
    // > 1, these are labeled differently to allow people to jointly consider the effects
    // of the array of SNPs, since we at the moment consider their effects only independently
    // (which has advantages for CADD, phyloP, phastCons, clinvar, etc reporting)
    if alleles.len() > 1 {
        return VariantEnum::Multi((MNP, positions, references, alleles));
    }

    if alleles[0].len() > 1 {
        if alleles[0][0] == b'-' {
            return VariantEnum::Multi((DEL, positions, references, alleles));
        }

        return VariantEnum::Multi((INS, positions, references, alleles));
    }

    panic!("WTF");
}

fn write_samples_type(
    header: &Vec<Vec<u8>>,
    samples: &Vec<u32>,
    buffer: &mut Vec<u8>,
    n_samples: f32,
    f_buf: &mut [u8; 15],
) {
    if samples.len() == 0 {
        buffer.push(b'!');
        buffer.push(b'\t');
        buffer.push(b'0');
        return;
    }

    for (idx, sample_idx) in samples.iter().enumerate() {
        buffer.extend_from_slice(&header[*sample_idx as usize]);

        if idx < samples.len() - 1 {
            buffer.push(b';');
        }
    }

    buffer.push(b'\t');

    write_f32(buffer, samples.len() as f32 / n_samples, f_buf);
}

fn write_ac_an(buffer: &mut Vec<u8>, ac: u32, an: u32, bytes: &mut Vec<u8>, f_buf: &mut [u8; 15]) {
    if ac == 0 {
        buffer.push(b'0');
        buffer.push(b'\t');
    } else {
        write_int(buffer, ac, bytes);
        buffer.push(b'\t');
    }
    write_int(buffer, an, bytes);
    buffer.push(b'\t');
    write_f32(buffer, ac as f32 / an as f32, f_buf);
}

fn write_int<T: itoa::Integer>(buffer: &mut Vec<u8>, val: T, mut b: &mut Vec<u8>) {
    itoa::write(&mut b, val).unwrap();
    buffer.extend_from_slice(&b);
    b.clear();
}

fn write_f32(buffer: &mut Vec<u8>, val: f32, f_buf: &mut [u8; 15]) {
    unsafe {
        let n = ryu::raw::f2s_buffered_n(val, &mut f_buf[0]);
        buffer.extend_from_slice(&f_buf[0..n]);
    };
}

fn write_samples(
    header: &Vec<Vec<u8>>,
    buffer: &mut Vec<u8>,
    hets: &Vec<u32>,
    homs: &Vec<u32>,
    missing_buffer: &Vec<u8>,
    effective_samples: f32,
    ac: u32,
    an: u32,
    bytes: &mut Vec<u8>,
    f_buf: &mut [u8; 15],
) {
    buffer.push(b'\t');

    write_samples_type(header, homs, buffer, effective_samples, f_buf);

    buffer.push(b'\t');

    write_samples_type(header, hets, buffer, effective_samples, f_buf);

    buffer.push(b'\t');

    buffer.extend_from_slice(&missing_buffer);

    buffer.push(b'\t');

    write_ac_an(buffer, ac, an, bytes, f_buf);
}

fn process_lines(header: &Vec<Vec<u8>>, rows: Vec<Vec<u8>>) -> usize {
    let n_samples = header.len() - 9;

    let mut homs: Vec<Vec<u32>> = Vec::new();
    let mut hets: Vec<Vec<u32>> = Vec::new();

    // Even in multiallelic case, missing in one means missing in all
    let mut missing: Vec<u32> = Vec::new();
    let mut ac: Vec<u32> = Vec::new();
    let mut an: u32 = 0;

    let mut allowed_filters: HashMap<&[u8], bool> = HashMap::new();
    allowed_filters.insert(b".", true);
    allowed_filters.insert(b"PASS", true);

    let excluded_filters: HashMap<&[u8], bool> = HashMap::new();
    let mut n_count = 0;
    let mut simple_gt = false;

    let mut chrom: &[u8] = b"";
    let mut pos: &[u8] = b"";
    let mut refr: &[u8] = b"";
    let mut alt: &[u8] = b"";

    let mut gt_range: &[u8];
    let mut buffer = Vec::with_capacity(100_000);

    let mut writer = std::io::stdout();

    let mut bytes = Vec::new();
    let mut f_buf: [u8; 15];
    unsafe {
        f_buf = std::mem::uninitialized();
    }

    let mut effective_samples: f32;
    let mut missing_buffer: Vec<u8> = Vec::new();
    let mut found_ac = 0;
    for row in rows.iter() {
        found_ac = 0;

        let mut alleles: VariantEnum = VariantEnum::None;

        'field_loop: for (idx, field) in row.split(|byt| *byt == b'\t').enumerate() {
            if idx == CHROM_IDX {
                chrom = field;
                continue;
            }

            if idx == POS_IDX {
                pos = field;
                continue;
            }

            if idx == REF_IDX {
                refr = field;
                continue;
            }

            if idx == ALT_IDX {
                alt = field;
                continue;
            }

            if idx == FILTER_IDX {
                if !filter_passes(field, &allowed_filters, &excluded_filters) {
                    break;
                }

                alleles = get_alleles(pos, refr, alt);

                match &alleles {
                    VariantEnum::Multi(v) => {
                        found_ac = v.3.len();
                    }
                    VariantEnum::None => {
                        // TODO: LOG
                        break;
                    }
                    _ => {
                        found_ac = 1;
                    }
                }

                continue;
            }

            if n_samples == 0 {
                break;
            }

            if idx == FORMAT_IDX {
                an = 0;

                simple_gt = memchr(b':', field) == None;
                missing.clear();
                hets = vec![Vec::new(); found_ac];
                homs = vec![Vec::new(); found_ac];
                ac = vec![0; found_ac];

                continue;
            }

            if idx > FORMAT_IDX {
                // TODO: Check quality if available
                if field.len() >= 3 && (field[1] == b'|' || field[1] == b'/') {
                    if field.len() == 3 || field[3] == b':' {
                        if field[0] == b'.' || field[2] == b'.' {
                            missing.push(idx as u32);
                            continue;
                        }

                        if field[0] == b'0' && field[2] == b'0' {
                            an += 2;
                            continue;
                        }

                        if field[0] == b'1' && field[2] == b'0'
                            || (field[0] == b'0' && field[2] == b'1')
                        {
                            an += 2;
                            ac[0] += 1;
                            hets[0].push(idx as u32);

                            continue;
                        }

                        if field[0] == b'1' && field[2] == b'1' {
                            an += 2;
                            ac[0] += 2;
                            homs[0].push(idx as u32);

                            continue;
                        }
                    }
                }

                if simple_gt {
                    gt_range = field;
                } else {
                    // TODO: Don't rely on format?
                    let end = memchr(b':', field).unwrap();
                    gt_range = &field[0..end];
                }

                if memchr(b'.', gt_range) != None {
                    missing.push(idx as u32);
                    continue 'field_loop;
                }

                for gt in gt_range.split(|byt| *byt == b'|' || *byt == b'/') {
                    let gtn = usize::from_radix_10(gt);

                    if gtn.1 == 0 {
                        // TODO: Handle failure
                        panic!("WTF");
                    }

                    an += 1;

                    if gtn.0 == 0 {
                        continue;
                    }

                    // TODO: What do for complex or malformed alleles that we skipped?
                    // Currently we're counting against an
                    // Count genotype? no?

                    if gtn.0 > found_ac {
                        continue;
                    }

                    ac[{ gtn.0 - 1 }] += 1;
                }
            }
        }

        n_count += 1;

        if n_samples > 0 && an == 0 {
            continue;
        }

        if missing_buffer.len() > 0 {
            missing_buffer.clear()
        }

        write_samples_type(
            &header,
            &missing,
            &mut missing_buffer,
            n_samples as f32,
            &mut f_buf,
        );

        if missing.len() > 0 {
            effective_samples = { n_samples - missing.len() } as f32;
        } else {
            effective_samples = n_samples as f32;
        }

        match alleles {
            VariantEnum::Multi(v) => {
                let (site_type, t_pos, t_refr, t_alt) = v;

                for i in 0..t_pos.len() {
                    if n_samples > 0 && ac[i] == 0 {
                        continue;
                    }

                    if i > 0 {
                        buffer.push(b'\n');
                    }

                    if chrom[0] != b'c' {
                        buffer.extend_from_slice(b"chr");
                    }

                    buffer.extend_from_slice(&chrom);
                    buffer.push(b'\t');

                    itoa::write(&mut bytes, t_pos[i]).unwrap();
                    buffer.extend_from_slice(&bytes);
                    bytes.clear();

                    buffer.push(b'\t');

                    buffer.extend_from_slice(site_type);
                    buffer.push(b'\t');
                    buffer.push(*t_refr[i]);
                    buffer.push(b'\t');
                    buffer.extend_from_slice(&t_alt[i]);
                    buffer.push(b'\t');
                    buffer.push(NOT_TSTV);
                    buffer.push(b'\t');

                    write_samples(
                        header,
                        &mut buffer,
                        &hets[i],
                        &homs[i],
                        &missing_buffer,
                        effective_samples,
                        ac[i],
                        an,
                        &mut bytes,
                        &mut f_buf,
                    );
                }
            }
            VariantEnum::Snp(v) => {
                let (pos, refr, alt) = v;

                if n_samples > 0 && ac[0] == 0 {
                    continue;
                }

                if chrom[0] != b'c' {
                    buffer.extend_from_slice(b"chr");
                }

                buffer.extend_from_slice(&chrom);
                buffer.push(b'\t');

                buffer.extend_from_slice(pos);
                buffer.push(b'\t');
                buffer.extend_from_slice(SNP);
                buffer.push(b'\t');
                buffer.push(refr);
                buffer.push(b'\t');
                buffer.push(alt);
                buffer.push(b'\t');
                buffer.push(TSTV[refr as usize][alt as usize]);
                buffer.push(b'\t');

                write_samples(
                    header,
                    &mut buffer,
                    &hets[0],
                    &homs[0],
                    &missing_buffer,
                    effective_samples,
                    ac[0],
                    an,
                    &mut bytes,
                    &mut f_buf,
                );
            }
            VariantEnum::Del(v) => {
                let (pos, refr, alt) = v;

                if chrom[0] != b'c' {
                    buffer.extend_from_slice(b"chr");
                }

                buffer.extend_from_slice(&chrom);
                buffer.push(b'\t');

                write_int(&mut buffer, pos, &mut bytes);
                buffer.push(b'\t');
                buffer.extend_from_slice(DEL);
                buffer.push(b'\t');
                buffer.push(refr);
                buffer.push(b'\t');
                write_int(&mut buffer, alt, &mut bytes);
                buffer.push(b'\t');
                buffer.push(NOT_TSTV);
                buffer.push(b'\t');

                write_samples(
                    header,
                    &mut buffer,
                    &hets[0],
                    &homs[0],
                    &missing_buffer,
                    effective_samples,
                    ac[0],
                    an,
                    &mut bytes,
                    &mut f_buf,
                );
            }
            VariantEnum::Ins(v) => {
                let (pos, refr, alt) = v;

                if chrom[0] != b'c' {
                    buffer.extend_from_slice(b"chr");
                }

                buffer.extend_from_slice(&chrom);
                buffer.push(b'\t');

                buffer.extend_from_slice(pos);
                buffer.push(b'\t');
                buffer.extend_from_slice(INS);
                buffer.push(b'\t');
                buffer.push(refr);
                buffer.push(b'\t');
                buffer.extend_from_slice(alt);
                buffer.push(b'\t');
                buffer.push(NOT_TSTV);
                buffer.push(b'\t');

                write_samples(
                    header,
                    &mut buffer,
                    &hets[0],
                    &homs[0],
                    &missing_buffer,
                    effective_samples,
                    ac[0],
                    an,
                    &mut bytes,
                    &mut f_buf,
                );
            }
            VariantEnum::None => {
                continue;
            }
        }

        buffer.push(b'\n');
    }

    writer.write_all(&buffer).unwrap();

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

    let header: Arc<Vec<Vec<u8>>> = Arc::new(
        get_header(&mut stdin_lock)
            .split(|b| *b == b'\t')
            .map(|sample| Vec::from(sample))
            .collect(),
    );

    if header.len() == 9 {
        info!("Found 9 header fields. When genotypes present, we expect 1+ samples after FORMAT (10 fields minimum)")
    }

    for _i in 0..n_cpus {
        let r = r1.clone();
        let s = s2.clone();
        let header = Arc::clone(&header);

        thread::spawn(move || {
            let mut n_count: usize = 0;

            loop {
                let message = match r.recv() {
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

        if thread_completed == n_cpus {
            break;
        }
    }

    assert_eq!(total, n_count);

    return Ok(());
}
