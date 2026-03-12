use anyhow::{Context, Result};
use std::collections::HashMap;

// ── 2벌식 키보드 매핑 테이블 ─────────────────────────────────

const SHIFT: &str = "[SHIFT]";
const BLANK: &str = "[BLANK]";

const INITIALS: [&str; 19] = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
];

const MEDIALS: [&str; 21] = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
    "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
];

const FINALS: [Option<&str>; 28] = [
    None, Some("ㄱ"), Some("ㄲ"), Some("ㄳ"), Some("ㄴ"), Some("ㄵ"),
    Some("ㄶ"), Some("ㄷ"), Some("ㄹ"), Some("ㄺ"), Some("ㄻ"),
    Some("ㄼ"), Some("ㄽ"), Some("ㄾ"), Some("ㄿ"), Some("ㅀ"),
    Some("ㅁ"), Some("ㅂ"), Some("ㅄ"), Some("ㅅ"), Some("ㅆ"),
    Some("ㅇ"), Some("ㅈ"), Some("ㅊ"), Some("ㅋ"), Some("ㅌ"),
    Some("ㅍ"), Some("ㅎ"),
];

fn is_hangul_syllable(ch: char) -> bool {
    ('\u{AC00}'..='\u{D7A3}').contains(&ch)
}

fn is_consonant(s: &str) -> bool {
    matches!(s,
        "ㄱ"|"ㄲ"|"ㄳ"|"ㄴ"|"ㄵ"|"ㄶ"|"ㄷ"|"ㄸ"|"ㄹ"|"ㄺ"|"ㄻ"|"ㄼ"|"ㄽ"|"ㄾ"|"ㄿ"|"ㅀ"|
        "ㅁ"|"ㅂ"|"ㅃ"|"ㅄ"|"ㅅ"|"ㅆ"|"ㅇ"|"ㅈ"|"ㅉ"|"ㅊ"|"ㅋ"|"ㅌ"|"ㅍ"|"ㅎ"
    )
}

fn is_vowel(s: &str) -> bool {
    matches!(s,
        "ㅏ"|"ㅐ"|"ㅑ"|"ㅒ"|"ㅓ"|"ㅔ"|"ㅕ"|"ㅖ"|"ㅗ"|"ㅘ"|"ㅙ"|"ㅚ"|"ㅛ"|
        "ㅜ"|"ㅝ"|"ㅞ"|"ㅟ"|"ㅠ"|"ㅡ"|"ㅢ"|"ㅣ"
    )
}

fn is_basic_consonant(s: &str) -> bool {
    matches!(s, "ㄱ"|"ㄴ"|"ㄷ"|"ㄹ"|"ㅁ"|"ㅂ"|"ㅅ"|"ㅇ"|"ㅈ"|"ㅊ"|"ㅋ"|"ㅌ"|"ㅍ"|"ㅎ")
}

fn is_jamo(s: &str) -> bool {
    is_consonant(s) || is_vowel(s)
}

fn decompose_syllable(ch: char) -> (usize, usize, usize) {
    let code = ch as u32 - 0xAC00;
    let initial = (code / 588) as usize;
    let medial = ((code % 588) / 28) as usize;
    let final_ = (code % 28) as usize;
    (initial, medial, final_)
}

fn compose_syllable(ini: usize, med: usize, fin: usize) -> char {
    char::from_u32(0xAC00 + (ini * 588 + med * 28 + fin) as u32).unwrap()
}

// ── 자모 분해 매핑 ──────────────────────────────────

fn double_consonant_base(jamo: &str) -> Option<&'static str> {
    match jamo {
        "ㄲ" => Some("ㄱ"), "ㄸ" => Some("ㄷ"), "ㅃ" => Some("ㅂ"),
        "ㅆ" => Some("ㅅ"), "ㅉ" => Some("ㅈ"),
        _ => None,
    }
}

fn compound_final_parts(jamo: &str) -> Option<(&'static str, &'static str)> {
    match jamo {
        "ㄳ" => Some(("ㄱ", "ㅅ")), "ㄵ" => Some(("ㄴ", "ㅈ")), "ㄶ" => Some(("ㄴ", "ㅎ")),
        "ㄺ" => Some(("ㄹ", "ㄱ")), "ㄻ" => Some(("ㄹ", "ㅁ")), "ㄼ" => Some(("ㄹ", "ㅂ")),
        "ㄽ" => Some(("ㄹ", "ㅅ")), "ㄾ" => Some(("ㄹ", "ㅌ")), "ㄿ" => Some(("ㄹ", "ㅍ")),
        "ㅀ" => Some(("ㄹ", "ㅎ")), "ㅄ" => Some(("ㅂ", "ㅅ")),
        _ => None,
    }
}

fn compound_vowel_parts(jamo: &str) -> Option<(&'static str, &'static str)> {
    match jamo {
        "ㅘ" => Some(("ㅗ", "ㅏ")), "ㅙ" => Some(("ㅗ", "ㅐ")), "ㅚ" => Some(("ㅗ", "ㅣ")),
        "ㅝ" => Some(("ㅜ", "ㅓ")), "ㅞ" => Some(("ㅜ", "ㅔ")), "ㅟ" => Some(("ㅜ", "ㅣ")),
        "ㅢ" => Some(("ㅡ", "ㅣ")),
        _ => None,
    }
}

fn shift_vowel_base(jamo: &str) -> Option<&'static str> {
    match jamo {
        "ㅒ" => Some("ㅐ"), "ㅖ" => Some("ㅔ"),
        _ => None,
    }
}

fn shift_consonant_reverse(base: &str) -> Option<&'static str> {
    match base {
        "ㄱ" => Some("ㄲ"), "ㄷ" => Some("ㄸ"), "ㅂ" => Some("ㅃ"),
        "ㅅ" => Some("ㅆ"), "ㅈ" => Some("ㅉ"),
        _ => None,
    }
}

fn shift_vowel_reverse(base: &str) -> Option<&'static str> {
    match base {
        "ㅐ" => Some("ㅒ"), "ㅔ" => Some("ㅖ"),
        _ => None,
    }
}

fn compound_vowel_reverse(a: &str, b: &str) -> Option<&'static str> {
    match (a, b) {
        ("ㅗ", "ㅏ") => Some("ㅘ"), ("ㅗ", "ㅐ") => Some("ㅙ"), ("ㅗ", "ㅣ") => Some("ㅚ"),
        ("ㅜ", "ㅓ") => Some("ㅝ"), ("ㅜ", "ㅔ") => Some("ㅞ"), ("ㅜ", "ㅣ") => Some("ㅟ"),
        ("ㅡ", "ㅣ") => Some("ㅢ"),
        _ => None,
    }
}

fn compound_final_reverse(a: &str, b: &str) -> Option<&'static str> {
    match (a, b) {
        ("ㄱ", "ㅅ") => Some("ㄳ"), ("ㄴ", "ㅈ") => Some("ㄵ"), ("ㄴ", "ㅎ") => Some("ㄶ"),
        ("ㄹ", "ㄱ") => Some("ㄺ"), ("ㄹ", "ㅁ") => Some("ㄻ"), ("ㄹ", "ㅂ") => Some("ㄼ"),
        ("ㄹ", "ㅅ") => Some("ㄽ"), ("ㄹ", "ㅌ") => Some("ㄾ"), ("ㄹ", "ㅍ") => Some("ㄿ"),
        ("ㄹ", "ㅎ") => Some("ㅀ"), ("ㅂ", "ㅅ") => Some("ㅄ"),
        _ => None,
    }
}

fn initial_to_idx(s: &str) -> Option<usize> {
    INITIALS.iter().position(|&x| x == s)
}

fn medial_to_idx(s: &str) -> Option<usize> {
    MEDIALS.iter().position(|&x| x == s)
}

fn final_to_idx(s: &str) -> Option<usize> {
    FINALS.iter().position(|f| *f == Some(s))
}

// ── 자음 → 키보드 스트로크 ──────────────────────────

fn consonant_to_keystrokes(jamo: &str) -> Vec<String> {
    if let Some(base) = double_consonant_base(jamo) {
        return vec![SHIFT.to_string(), base.to_string()];
    }
    if let Some((a, b)) = compound_final_parts(jamo) {
        let mut r = consonant_to_keystrokes(a);
        r.extend(consonant_to_keystrokes(b));
        return r;
    }
    vec![jamo.to_string()]
}

fn vowel_to_keystrokes(jamo: &str) -> Vec<String> {
    if let Some(base) = shift_vowel_base(jamo) {
        return vec![SHIFT.to_string(), base.to_string()];
    }
    if let Some((a, b)) = compound_vowel_parts(jamo) {
        let mut r = vowel_to_keystrokes(a);
        r.extend(vowel_to_keystrokes(b));
        return r;
    }
    vec![jamo.to_string()]
}

// ── 전처리: 텍스트 → 키보드 스트로크 ─────────────────

fn needs_blank_between(prev: char, next_s: &str) -> bool {
    let prev_s = prev.to_string();
    let prev_syl = is_hangul_syllable(prev);
    let prev_jamo = is_jamo(&prev_s);
    if !is_jamo(next_s) { return false; }
    if !(prev_syl || prev_jamo) { return false; }

    let next_is_consonant = is_consonant(next_s);
    let next_is_vowel = is_vowel(next_s);

    if prev_syl {
        let (_, med_idx, fin_idx) = decompose_syllable(prev);
        let has_final = fin_idx != 0;
        if !has_final {
            if next_is_consonant { return true; }
            if next_is_vowel {
                let med = MEDIALS[med_idx];
                let next_keys = vowel_to_keystrokes(next_s);
                return compound_vowel_reverse(med, &next_keys[0]).is_some();
            }
        } else {
            let fin = FINALS[fin_idx].unwrap();
            if next_is_vowel { return true; }
            if next_is_consonant {
                let fin_keys = consonant_to_keystrokes(fin);
                let last_fin = &fin_keys[fin_keys.len() - 1];
                let next_keys = consonant_to_keystrokes(next_s);
                if next_keys[0] == SHIFT { return false; }
                return compound_final_reverse(last_fin, &next_keys[0]).is_some();
            }
        }
    } else if prev_jamo {
        if is_consonant(&prev_s) && next_is_vowel { return true; }
        if is_vowel(&prev_s) && next_is_vowel {
            let prev_keys = vowel_to_keystrokes(&prev_s);
            let next_keys = vowel_to_keystrokes(next_s);
            let last_prev = &prev_keys[prev_keys.len() - 1];
            return compound_vowel_reverse(last_prev, &next_keys[0]).is_some();
        }
    }
    false
}

/// 텍스트를 2벌식 키보드 스트로크 토큰 리스트로 변환
pub fn preprocess(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();

    for (i, &ch) in chars.iter().enumerate() {
        let ch_s = ch.to_string();

        // BLANK 삽입 판단
        if i > 0 && is_jamo(&ch_s) && needs_blank_between(chars[i - 1], &ch_s) {
            tokens.push(BLANK.to_string());
        }

        if is_hangul_syllable(ch) {
            let (ini_idx, med_idx, fin_idx) = decompose_syllable(ch);
            tokens.extend(consonant_to_keystrokes(INITIALS[ini_idx]));
            tokens.extend(vowel_to_keystrokes(MEDIALS[med_idx]));
            if let Some(fin) = FINALS[fin_idx] {
                tokens.extend(consonant_to_keystrokes(fin));
            }
        } else if is_consonant(&ch_s) {
            tokens.extend(consonant_to_keystrokes(&ch_s));
        } else if is_vowel(&ch_s) {
            tokens.extend(vowel_to_keystrokes(&ch_s));
        } else {
            tokens.push(ch_s);
        }
    }
    tokens
}

/// 키보드 스트로크 토큰 리스트를 텍스트로 재합성
pub fn postprocess(tokens: &[String]) -> String {
    let mut result = Vec::new();

    // 1단계: SHIFT 토큰 해석
    let mut expanded = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == SHIFT && i + 1 < tokens.len() {
            let next = &tokens[i + 1];
            if let Some(double) = shift_consonant_reverse(next) {
                expanded.push(double.to_string());
                i += 2;
                continue;
            }
            if let Some(double) = shift_vowel_reverse(next) {
                expanded.push(double.to_string());
                i += 2;
                continue;
            }
            expanded.push(tokens[i].clone());
            expanded.push(tokens[i + 1].clone());
            i += 2;
        } else {
            expanded.push(tokens[i].clone());
            i += 1;
        }
    }

    // 2단계: IME 시뮬레이션
    let mut ini_idx: Option<usize> = None;
    let mut med_idx: Option<usize> = None;
    let mut fin: Option<String> = None;
    let mut fin_idx: Option<usize> = None;

    let flush = |ini: &mut Option<usize>, med: &mut Option<usize>,
                 f: &mut Option<String>, fi: &mut Option<usize>,
                 result: &mut Vec<String>| {
        if let (Some(ini_v), Some(med_v)) = (*ini, *med) {
            result.push(compose_syllable(ini_v, med_v, fi.unwrap_or(0)).to_string());
        } else if let Some(ini_v) = *ini {
            result.push(INITIALS[ini_v].to_string());
        } else if let Some(med_v) = *med {
            result.push(MEDIALS[med_v].to_string());
        }
        *ini = None; *med = None; *f = None; *fi = None;
    };

    for tok in &expanded {
        if tok == BLANK {
            flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
            continue;
        }

        let tok_is_consonant = is_consonant(tok) && initial_to_idx(tok).is_some();
        let tok_is_vowel = is_vowel(tok) && medial_to_idx(tok).is_some();

        if tok_is_consonant {
            if ini_idx.is_none() && med_idx.is_none() {
                ini_idx = initial_to_idx(tok);
            } else if ini_idx.is_some() && med_idx.is_none() {
                flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                ini_idx = initial_to_idx(tok);
            } else if ini_idx.is_some() && med_idx.is_some() && fin.is_none() {
                if let Some(fi) = final_to_idx(tok) {
                    fin = Some(tok.clone());
                    fin_idx = Some(fi);
                } else {
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    ini_idx = initial_to_idx(tok);
                }
            } else if fin.is_some() {
                let compound = compound_final_reverse(fin.as_ref().unwrap(), tok);
                if let Some(cf) = compound {
                    if let Some(cfi) = final_to_idx(cf) {
                        fin = Some(cf.to_string());
                        fin_idx = Some(cfi);
                    } else {
                        flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                        ini_idx = initial_to_idx(tok);
                    }
                } else {
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    ini_idx = initial_to_idx(tok);
                }
            } else {
                flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                ini_idx = initial_to_idx(tok);
            }
        } else if tok_is_vowel {
            if ini_idx.is_some() && med_idx.is_some() && fin.is_some() {
                // 종성 도둑: 종성을 떼어서 다음 초성으로
                let fin_str = fin.as_ref().unwrap().clone();
                if let Some((part_a, part_b)) = compound_final_parts(&fin_str) {
                    fin = Some(part_a.to_string());
                    fin_idx = final_to_idx(part_a);
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    ini_idx = initial_to_idx(part_b);
                } else {
                    let carry = fin_str;
                    fin = None;
                    fin_idx = None;
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    ini_idx = initial_to_idx(&carry);
                }
                med_idx = medial_to_idx(tok);
            } else if ini_idx.is_some() && med_idx.is_some() && fin.is_none() {
                if let Some(cv) = compound_vowel_reverse(MEDIALS[med_idx.unwrap()], tok) {
                    med_idx = medial_to_idx(cv);
                } else {
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    med_idx = medial_to_idx(tok);
                }
            } else if ini_idx.is_some() && med_idx.is_none() {
                med_idx = medial_to_idx(tok);
            } else if med_idx.is_some() {
                if let Some(cv) = compound_vowel_reverse(MEDIALS[med_idx.unwrap()], tok) {
                    med_idx = medial_to_idx(cv);
                } else {
                    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
                    med_idx = medial_to_idx(tok);
                }
            } else {
                med_idx = medial_to_idx(tok);
            }
        } else {
            flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);
            result.push(tok.clone());
        }
    }
    flush(&mut ini_idx, &mut med_idx, &mut fin, &mut fin_idx, &mut result);

    result.join("")
}

// ── KeyboardTokenizer ────────────────────────────────

pub struct KeyboardTokenizer {
    bpe: tokenizers::Tokenizer,
    jamo_map: HashMap<String, u32>,
    id_to_jamo: HashMap<u32, String>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
}

impl KeyboardTokenizer {
    pub fn from_dir(dir: &str) -> Result<Self> {
        let tok_path = format!("{}/keyboard_tokenizer.json", dir);
        let jamo_path = format!("{}/jamo_token_map.json", dir);

        let bpe = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("토크나이저 로드 실패: {}", e))?;

        let jamo_data = std::fs::read_to_string(&jamo_path)
            .context("jamo_token_map.json 로드 실패")?;
        let jamo_map: HashMap<String, u32> = serde_json::from_str(&jamo_data)?;
        let id_to_jamo: HashMap<u32, String> = jamo_map.iter().map(|(k, v)| (*v, k.clone())).collect();

        let pad_id = bpe.token_to_id("[PAD]").unwrap_or(0);
        let bos_id = bpe.token_to_id("[BOS]").unwrap_or(2);
        let eos_id = bpe.token_to_id("[EOS]").unwrap_or(3);
        let unk_id = bpe.token_to_id("[UNK]").unwrap_or(1);

        Ok(Self { bpe, jamo_map, id_to_jamo, pad_id, bos_id, eos_id, unk_id })
    }

    pub fn vocab_size(&self) -> usize {
        self.bpe.get_vocab_size(true)
    }

    /// 텍스트 → 토큰 ID 리스트 (BOS/EOS 포함)
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let keystrokes = preprocess(text);
        let mut ids = vec![self.bos_id];
        let flush_buf = |buf: &mut String, ids: &mut Vec<u32>, bpe: &tokenizers::Tokenizer| {
            if !buf.is_empty() {
                if let Ok(enc) = bpe.encode(buf.as_str(), false) {
                    ids.extend(enc.get_ids());
                }
                buf.clear();
            }
        };

        let mut buf = String::new();
        for tok in &keystrokes {
            if let Some(&id) = self.jamo_map.get(tok.as_str()) {
                flush_buf(&mut buf, &mut ids, &self.bpe);
                ids.push(id);
            } else {
                buf.push_str(tok);
            }
        }
        flush_buf(&mut buf, &mut ids, &self.bpe);
        ids.push(self.eos_id);
        ids
    }

    /// 토큰 ID 리스트 → 텍스트
    pub fn decode(&self, ids: &[u32]) -> String {
        let special = [self.pad_id, self.bos_id, self.eos_id, self.unk_id];
        let filtered: Vec<u32> = ids.iter().copied().filter(|id| !special.contains(id)).collect();

        let mut keystrokes = Vec::new();
        let mut non_jamo_ids = Vec::new();

        let flush_ids = |nj_ids: &mut Vec<u32>, ks: &mut Vec<String>, bpe: &tokenizers::Tokenizer| {
            if !nj_ids.is_empty() {
                if let Ok(decoded) = bpe.decode(nj_ids, false) {
                    for ch in decoded.chars() {
                        ks.push(ch.to_string());
                    }
                }
                nj_ids.clear();
            }
        };

        for &tid in &filtered {
            if let Some(jamo) = self.id_to_jamo.get(&tid) {
                flush_ids(&mut non_jamo_ids, &mut keystrokes, &self.bpe);
                keystrokes.push(jamo.clone());
            } else {
                non_jamo_ids.push(tid);
            }
        }
        flush_ids(&mut non_jamo_ids, &mut keystrokes, &self.bpe);

        postprocess(&keystrokes)
    }
}
