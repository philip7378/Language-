// ============================================================================
// RCL v1.0 Monolithic Boot Kernel (Phase 8: Module System & Streaming)
// ============================================================================

use std::io::{self, BufRead, Write};
// Embed the pre-compiled standard library directly into the executable.
// This makes the 'rcs' binary a completely self-contained language.
const CORE_SOURCE: &str = include_str!("core.rcl");
mod algebra {
    use std::collections::BTreeMap;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Complex64 { pub re: f64, pub im: f64 }

    impl Complex64 {
        pub const fn new(re: f64, im: f64) -> Self { Self { re, im } }
        pub const fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
        pub const fn one() -> Self { Self { re: 1.0, im: 0.0 } }
        pub const fn real(r: f64) -> Self { Self { re: r, im: 0.0 } }
        pub fn conjugate(&self) -> Self { Self { re: self.re, im: -self.im } }
        pub fn norm_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
        pub fn modulus(&self) -> f64 { self.norm_sq().sqrt() }
    }

    impl std::ops::Add for Complex64 { type Output = Self; fn add(self, r: Self) -> Self { Self { re: self.re + r.re, im: self.im + r.im } } }
    impl std::ops::Sub for Complex64 { type Output = Self; fn sub(self, r: Self) -> Self { Self { re: self.re - r.re, im: self.im - r.im } } }
    impl std::ops::Mul for Complex64 { type Output = Self; fn mul(self, r: Self) -> Self { Self { re: self.re * r.re - self.im * r.im, im: self.re * r.im + self.im * r.re } } }
    impl std::ops::Mul<f64> for Complex64 { type Output = Self; fn mul(self, r: f64) -> Self { Self { re: self.re * r, im: self.im * r } } }
    impl std::ops::Neg for Complex64 { type Output = Self; fn neg(self) -> Self { Self { re: -self.re, im: -self.im } } }

    impl std::ops::Div for Complex64 {
        type Output = Self;
        fn div(self, r: Self) -> Self {
            let denom = r.norm_sq();
            if denom < 1e-30 { return Complex64::zero(); }
            Self { re: (self.re * r.re + self.im * r.im) / denom, im: (self.im * r.re - self.re * r.im) / denom }
        }
    }

    pub type Matrix = Vec<Vec<Complex64>>;
    pub type CVector = Vec<Complex64>;

    pub fn identity(n: usize) -> Matrix { let mut m = vec![vec![Complex64::zero(); n]; n]; for i in 0..n { m[i][i] = Complex64::one(); } m }
    pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix { let n = a.len(); if n == 0 || b.is_empty() { return Vec::new(); } let m = b[0].len(); let k = b.len(); let mut r = vec![vec![Complex64::zero(); m]; n]; for i in 0..n { for j in 0..m { for l in 0..k { r[i][j] = r[i][j] + a[i][l] * b[l][j]; } } } r }
    pub fn mat_add(a: &Matrix, b: &Matrix) -> Matrix { let n = a.len(); let m = a[0].len(); let mut r = vec![vec![Complex64::zero(); m]; n]; for i in 0..n { for j in 0..m { r[i][j] = a[i][j] + b[i][j]; } } r }
    pub fn mat_scale(a: &Matrix, s: Complex64) -> Matrix { a.iter().map(|row| row.iter().map(|&x| x * s).collect()).collect() }
    pub fn mat_vec(a: &Matrix, v: &CVector) -> CVector { let n = a.len(); let mut r = vec![Complex64::zero(); n]; for i in 0..n { for j in 0..v.len() { r[i] = r[i] + a[i][j] * v[j]; } } r }
    pub fn inner(u: &CVector, v: &CVector) -> Complex64 { let mut r = Complex64::zero(); for i in 0..u.len().min(v.len()) { r = r + u[i].conjugate() * v[i]; } r }
    pub fn norm_sq(v: &CVector) -> f64 { v.iter().map(|x| x.norm_sq()).sum() }

    pub fn null_space(m: &Matrix) -> Vec<CVector> {
        let n = m.len(); if n == 0 { return Vec::new(); }
        let mut a = m.clone(); let mut pivots: Vec<Option<usize>> = vec![None; n]; let mut row = 0;
        for col in 0..n {
            let mut pivot_row = None;
            for i in row..n { if a[i][col].norm_sq() > 1e-12 { pivot_row = Some(i); break; } }
            if let Some(p) = pivot_row {
                a.swap(row, p); let inv_pivot = Complex64::one() / a[row][col];
                for j in col..n { a[row][j] = a[row][j] * inv_pivot; }
                for i in 0..n { if i != row && a[i][col].norm_sq() > 1e-12 { let factor = a[i][col]; for j in col..n { a[i][j] = a[i][j] - factor * a[row][j]; } } }
                pivots[col] = Some(row); row += 1;
            }
        }
        let mut basis = Vec::new();
        for col in 0..n {
            if pivots[col].is_none() {
                let mut v = vec![Complex64::zero(); n]; v[col] = Complex64::one();
                for c in 0..n { if let Some(r) = pivots[c] { let mut s = Complex64::zero(); for j in c+1..n { s = s + a[r][j] * v[j]; } v[c] = -s; } }
                basis.push(v);
            }
        }
        basis
    }

    pub fn gram_schmidt(basis: &[CVector]) -> Vec<CVector> {
        let mut ortho = Vec::new();
        for v in basis {
            let mut u = v.clone();
            for e in &ortho { let p = inner(e, v); for i in 0..u.len() { u[i] = u[i] - p * e[i]; } }
            let norm = norm_sq(&u).sqrt();
            if norm > 1e-12 { let e: CVector = u.iter().map(|x| *x * (1.0 / norm)).collect(); ortho.push(e); }
        }
        ortho
    }

    pub fn generator_matrix(basis_dim: usize, seed: u64) -> Matrix {
        let mut m = vec![vec![Complex64::zero(); basis_dim]; basis_dim]; let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for i in 0..basis_dim { for j in i..basis_dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); let re = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); let im = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            let val = Complex64::new(re, im); m[i][j] = val; m[j][i] = val.conjugate();
        }}
        for i in 0..basis_dim { m[i][i].im = 0.0; } m
    }

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct GeneratorId(pub u64);

    #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Monomial { pub word: Vec<GeneratorId> }

    impl Monomial {
        pub fn unit() -> Self { Self { word: Vec::new() } }
        pub fn atom(g: GeneratorId) -> Self { Self { word: vec![g] } }
        pub fn compose(&self, o: &Self) -> Self { let mut w = self.word.clone(); w.extend(o.word.iter().cloned()); Self { word: w } }
        pub fn is_unit(&self) -> bool { self.word.is_empty() }
    }

    #[derive(Clone, Debug)]
    pub struct AlgebraElement { pub terms: BTreeMap<Monomial, Complex64> }

    impl AlgebraElement {
        pub fn unit() -> Self { let mut t = BTreeMap::new(); t.insert(Monomial::unit(), Complex64::one()); Self { terms: t } }
        pub fn generator(g: GeneratorId) -> Self { let mut t = BTreeMap::new(); t.insert(Monomial::atom(g), Complex64::one()); Self { terms: t } }
        pub fn zero() -> Self { Self { terms: BTreeMap::new() } }
        pub fn scalar(r: f64) -> Self { Self::unit().scale(Complex64::real(r)) }
        pub fn scale(&self, l: Complex64) -> Self { if l.norm_sq() < 1e-30 { return Self::zero(); } Self { terms: self.terms.iter().map(|(m, c)| (m.clone(), l * *c)).collect() } }
        pub fn add(&self, o: &Self) -> Self { let mut t = self.terms.clone(); for (m, c) in &o.terms { let e = t.entry(m.clone()).or_insert_with(Complex64::zero); *e = *e + *c; if e.norm_sq() < 1e-30 { t.remove(m); } } Self { terms: t } }
        pub fn sub(&self, o: &Self) -> Self { self.add(&o.scale(Complex64::real(-1.0))) }
        pub fn mul(&self, o: &Self) -> Self { let mut t: BTreeMap<Monomial, Complex64> = BTreeMap::new(); for (m1, c1) in &self.terms { for (m2, c2) in &o.terms { let m = m1.compose(m2); let e = t.entry(m).or_insert_with(Complex64::zero); *e = *e + *c1 * *c2; }} t.retain(|_, c| c.norm_sq() >= 1e-30); Self { terms: t } }
        pub fn as_scalar(&self) -> Option<f64> { if self.terms.is_empty() { return Some(0.0); } if self.terms.len() == 1 { if let Some((m, c)) = self.terms.iter().next() { if m.is_unit() && c.im.abs() < 1e-10 { return Some(c.re); } } } None }
        pub fn extract_token_sequence(&self) -> Option<Vec<GeneratorId>> { if self.terms.len() == 1 { if let Some((m, c)) = self.terms.iter().next() { if (c.re - 1.0).abs() < 1e-10 && c.im.abs() < 1e-10 { return Some(m.word.clone()); } } } None }
    }
}

mod ast {
    #[derive(Clone, Debug)]
    pub enum Decl {
        Generator { name: String, ty: String, init: Option<Expr> },
        Constraint { name: String, params: Vec<(String, String)>, condition: Expr },
        Event { name: String, params: Vec<(String, String)>, ret_type: Option<String>, requires: Vec<Expr>, body: Expr },
        Concept { name: String, tokens: Vec<String> },
        Import(String),
    }
    #[derive(Clone, Debug)] pub enum Stmt { Let { name: String, value: Expr }, Print(Expr), Ingest(String), Expr(Expr) }
    #[derive(Clone, Debug)] pub enum Expr { Int(i64), Float(f64), Str(String), Var(String), BinOp(Box<Expr>, BinOp, Box<Expr>), If(Box<Expr>, Box<Expr>, Box<Expr>), Call(String, Vec<Expr>), Match(Box<Expr>, Vec<MatchArm>), Block(Vec<Stmt>, Box<Expr>) }
    #[derive(Clone, Debug)] pub struct MatchArm { pub pattern: Expr, pub body: Expr, pub is_wildcard: bool }
    #[derive(Clone, Debug)] pub enum BinOp { Add, Sub, Mul, Div, Eq, NotEq, Gte, Lte, Gt, Lt }
    #[derive(Clone, Debug)] pub struct Program { pub decls: Vec<Decl>, pub stmts: Vec<Stmt> }
}

mod lexer {
    #[derive(Clone, Debug, PartialEq)]
    pub enum Tok { Gen, Let, Concept, Constraint, Event, Enforce, Requires, If, Else, Match, Ingest, Print, Fn, Import, Int(i64), Float(f64), Str(String), Ident(String), Plus, Minus, Star, Slash, Eq, NotEq, Gte, Lte, Gt, Lt, Assign, Arrow, FatArrow, Colon, Semicolon, Comma, LParen, RParen, LBrace, RBrace, Eof }
    #[derive(Clone, Debug)] pub struct Token { pub tok: Tok, pub line: usize, pub col: usize }
    pub struct Lexer<'a> { chars: std::iter::Peekable<std::str::Chars<'a>>, line: usize, col: usize }
    impl<'a> Lexer<'a> {
        pub fn new(src: &'a str) -> Self { Self { chars: src.chars().peekable(), line: 1, col: 1 } }
        fn peek(&mut self) -> Option<char> { self.chars.peek().copied() }
        fn advance(&mut self) -> Option<char> { let c = self.chars.next(); if c == Some('\n') { self.line += 1; self.col = 1; } else { self.col += 1; } c }
        fn skip_ws(&mut self) { while let Some(c) = self.peek() { if c.is_whitespace() { self.advance(); } else if c == '/' { let mut clone = self.chars.clone(); clone.next(); if clone.peek() == Some(&'/') { while let Some(c) = self.peek() { if c == '\n' { break; } self.advance(); } } else { break; } } else { break; } } }
        pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
            let mut tokens = Vec::new();
            loop { self.skip_ws(); let line = self.line; let col = self.col; let c = match self.peek() { Some(c) => c, None => { tokens.push(Token { tok: Tok::Eof, line, col }); break; } };
                let tok = match c {
                    '+' => { self.advance(); Tok::Plus } '-' => { self.advance(); if self.peek() == Some('>') { self.advance(); Tok::Arrow } else { Tok::Minus } }
                    '*' => { self.advance(); Tok::Star } '/' => { self.advance(); Tok::Slash }
                    '=' => { self.advance(); if self.peek() == Some('=') { self.advance(); Tok::Eq } else if self.peek() == Some('>') { self.advance(); Tok::FatArrow } else { Tok::Assign } }
                    '!' => { self.advance(); if self.peek() == Some('=') { self.advance(); Tok::NotEq } else { return Err(format!("{}:{}: Unexpected '!'", line, col)); } }
                    '>' => { self.advance(); if self.peek() == Some('=') { self.advance(); Tok::Gte } else { Tok::Gt } }
                    '<' => { self.advance(); if self.peek() == Some('=') { self.advance(); Tok::Lte } else { Tok::Lt } }
                    ':' => { self.advance(); Tok::Colon } ';' => { self.advance(); Tok::Semicolon } ',' => { self.advance(); Tok::Comma }
                    '(' => { self.advance(); Tok::LParen } ')' => { self.advance(); Tok::RParen } '{' => { self.advance(); Tok::LBrace } '}' => { self.advance(); Tok::RBrace }
                    '"' => { self.advance(); self.read_string(line, col)? } c if c.is_ascii_digit() => { self.read_number(line, col)? }
                    c if c.is_alphabetic() || c == '_' => { self.read_ident(line, col) } _ => return Err(format!("{}:{}: Unexpected character '{}'", line, col, c)),
                }; tokens.push(Token { tok, line, col }); }
            Ok(tokens)
        }
        fn read_string(&mut self, line: usize, col: usize) -> Result<Tok, String> { let mut s = String::new(); loop { match self.advance() { Some('"') => break, Some('\\') => { match self.advance() { Some('n') => s.push('\n'), Some('t') => s.push('\t'), Some('"') => s.push('"'), Some(c) => s.push(c), None => return Err(format!("{}:{}: Unterminated string", line, col)), } } Some(c) => s.push(c), None => return Err(format!("{}:{}: Unterminated string", line, col)), } } Ok(Tok::Str(s)) }
        fn read_number(&mut self, line: usize, col: usize) -> Result<Tok, String> { let mut s = String::new(); let mut is_float = false; while let Some(c) = self.peek() { if c.is_ascii_digit() { s.push(c); self.advance(); } else if c == '.' && !is_float { is_float = true; s.push(c); self.advance(); } else { break; } } if is_float { s.parse::<f64>().map(Tok::Float).map_err(|e| format!("{}:{}: Invalid float: {}", line, col, e)) } else { s.parse::<i64>().map(Tok::Int).map_err(|e| format!("{}:{}: Invalid int: {}", line, col, e)) } }
        fn read_ident(&mut self, _line: usize, _col: usize) -> Tok { let mut s = String::new(); while let Some(c) = self.peek() { if c.is_alphanumeric() || c == '_' { s.push(c); self.advance(); } else { break; } } match s.as_str() { "gen" => Tok::Gen, "let" => Tok::Let, "concept" => Tok::Concept, "constraint" => Tok::Constraint, "event" => Tok::Event, "enforce" => Tok::Enforce, "requires" => Tok::Requires, "if" => Tok::If, "else" => Tok::Else, "match" => Tok::Match, "ingest" => Tok::Ingest, "print" => Tok::Print, "fn" => Tok::Fn, "import" => Tok::Import, _ => Tok::Ident(s), } }
    }
}

mod parser {
    use super::ast::*; use super::lexer::{Tok, Token};
    pub struct Parser { tokens: Vec<Token>, pos: usize }
    impl Parser {
        pub fn new(tokens: Vec<Token>) -> Self { Self { tokens, pos: 0 } }
        fn peek(&self) -> &Tok { &self.tokens[self.pos].tok }
        fn advance(&mut self) -> Tok { let t = self.tokens[self.pos].tok.clone(); self.pos += 1; t }
        fn line(&self) -> usize { self.tokens[self.pos].line } fn col(&self) -> usize { self.tokens[self.pos].col }
        fn expect(&mut self, expected: &Tok) -> Result<(), String> { if self.peek() == expected { self.advance(); Ok(()) } else { Err(format!("{}:{}: Expected {:?}, got {:?}", self.line(), self.col(), expected, self.peek())) } }
        fn match_tok(&mut self, tok: &Tok) -> bool { if self.peek() == tok { self.advance(); true } else { false } }
        pub fn parse_program(&mut self) -> Result<Program, String> {
            let mut decls = Vec::new(); let mut stmts = Vec::new();
            while *self.peek() != Tok::Eof {
                while *self.peek() == Tok::Semicolon { self.advance(); }
                if *self.peek() == Tok::Eof { break; }
                match self.peek() {
                    Tok::Gen => decls.push(self.parse_gen()?),
                    Tok::Constraint => decls.push(self.parse_constraint()?),
                    Tok::Event => decls.push(self.parse_event()?),
                    Tok::Concept => decls.push(self.parse_concept()?),
                    Tok::Import => {
                        self.advance();
                        match self.advance() {
                            Tok::Str(s) => {
                                if *self.peek() == Tok::Semicolon { self.advance(); }
                                decls.push(Decl::Import(s));
                            }
                            other => return Err(format!("Expected string after import, got {:?}", other)),
                        }
                    }
                    Tok::Let => stmts.push(self.parse_let()?),
                    Tok::Print => stmts.push(self.parse_print()?),
                    Tok::Ingest => stmts.push(self.parse_ingest()?),
                    _ => stmts.push(Stmt::Expr(self.parse_expr()?)),
                }
                if *self.peek() == Tok::Semicolon { self.advance(); }
            }
            Ok(Program { decls, stmts })
        }
        fn parse_gen(&mut self) -> Result<Decl, String> { self.advance(); let name = self.parse_ident()?; self.expect(&Tok::Colon)?; let ty = self.parse_ident()?; let init = if self.match_tok(&Tok::Assign) { Some(self.parse_expr()?) } else { None }; self.expect(&Tok::Semicolon)?; Ok(Decl::Generator { name, ty, init }) }
        fn parse_constraint(&mut self) -> Result<Decl, String> { self.advance(); let name = self.parse_ident()?; self.expect(&Tok::LParen)?; let params = self.parse_params()?; self.expect(&Tok::RParen)?; self.expect(&Tok::LBrace)?; self.expect(&Tok::Enforce)?; let condition = self.parse_expr()?; self.expect(&Tok::Semicolon)?; self.expect(&Tok::RBrace)?; Ok(Decl::Constraint { name, params, condition }) }
        fn parse_event(&mut self) -> Result<Decl, String> { self.advance(); let name = self.parse_ident()?; self.expect(&Tok::LParen)?; let params = self.parse_params()?; self.expect(&Tok::RParen)?; let ret_type = if self.match_tok(&Tok::Arrow) { Some(self.parse_ident()?) } else { None }; let mut requires = Vec::new(); if self.match_tok(&Tok::Requires) { requires.push(self.parse_expr()?); while self.match_tok(&Tok::Comma) { requires.push(self.parse_expr()?); } } let body = self.parse_block()?; Ok(Decl::Event { name, params, ret_type, requires, body }) }
        fn parse_concept(&mut self) -> Result<Decl, String> { self.advance(); let name = self.parse_ident()?; self.expect(&Tok::Assign)?; self.expect(&Tok::LBrace)?; let mut tokens = Vec::new(); if *self.peek() != Tok::RBrace { if let Tok::Str(s) = self.advance() { tokens.push(s); } while self.match_tok(&Tok::Comma) { if let Tok::Str(s) = self.advance() { tokens.push(s); } else { return Err(format!("{}:{}: Expected string", self.line(), self.col())); } } } self.expect(&Tok::RBrace)?; self.expect(&Tok::Semicolon)?; Ok(Decl::Concept { name, tokens }) }
        fn parse_params(&mut self) -> Result<Vec<(String, String)>, String> { let mut params = Vec::new(); if *self.peek() != Tok::RParen { loop { let name = self.parse_ident()?; self.expect(&Tok::Colon)?; let ty = self.parse_ident()?; params.push((name, ty)); if !self.match_tok(&Tok::Comma) { break; } } } Ok(params) }
        fn parse_let(&mut self) -> Result<Stmt, String> { self.advance(); let name = self.parse_ident()?; self.expect(&Tok::Assign)?; let value = self.parse_expr()?; Ok(Stmt::Let { name, value }) }
        fn parse_print(&mut self) -> Result<Stmt, String> { self.advance(); self.expect(&Tok::LParen)?; let e = self.parse_expr()?; self.expect(&Tok::RParen)?; Ok(Stmt::Print(e)) }
        fn parse_ingest(&mut self) -> Result<Stmt, String> { self.advance(); match self.advance() { Tok::Str(s) => Ok(Stmt::Ingest(s)), other => Err(format!("Expected string after ingest, got {:?}", other)), } }
        fn parse_ident(&mut self) -> Result<String, String> { match self.advance() { Tok::Ident(s) => Ok(s), other => Err(format!("{}:{}: Expected identifier, got {:?}", self.line(), self.col(), other)), } }
        fn parse_block(&mut self) -> Result<Expr, String> { self.expect(&Tok::LBrace)?; let mut stmts = Vec::new(); let mut last_expr = Expr::Int(0); while *self.peek() != Tok::RBrace && *self.peek() != Tok::Eof { match self.peek() { Tok::Let => stmts.push(self.parse_let()?), Tok::Print => stmts.push(self.parse_print()?), Tok::Ingest => stmts.push(self.parse_ingest()?), _ => { let e = self.parse_expr()?; if *self.peek() == Tok::Semicolon { stmts.push(Stmt::Expr(e)); } else { last_expr = e; break; } } } while *self.peek() == Tok::Semicolon { self.advance(); } } self.expect(&Tok::RBrace)?; Ok(Expr::Block(stmts, Box::new(last_expr))) }
        fn parse_expr(&mut self) -> Result<Expr, String> { self.parse_comparison() }
        fn parse_comparison(&mut self) -> Result<Expr, String> { let mut left = self.parse_additive()?; loop { let op = match self.peek() { Tok::Eq => BinOp::Eq, Tok::NotEq => BinOp::NotEq, Tok::Gte => BinOp::Gte, Tok::Lte => BinOp::Lte, Tok::Gt => BinOp::Gt, Tok::Lt => BinOp::Lt, _ => return Ok(left), }; self.advance(); let right = self.parse_additive()?; left = Expr::BinOp(Box::new(left), op, Box::new(right)); } }
        fn parse_additive(&mut self) -> Result<Expr, String> { let mut left = self.parse_multiplicative()?; loop { let op = match self.peek() { Tok::Plus => BinOp::Add, Tok::Minus => BinOp::Sub, _ => return Ok(left), }; self.advance(); let right = self.parse_multiplicative()?; left = Expr::BinOp(Box::new(left), op, Box::new(right)); } }
        fn parse_multiplicative(&mut self) -> Result<Expr, String> { let mut left = self.parse_unary()?; loop { let op = match self.peek() { Tok::Star => BinOp::Mul, Tok::Slash => BinOp::Div, _ => return Ok(left), }; self.advance(); let right = self.parse_unary()?; left = Expr::BinOp(Box::new(left), op, Box::new(right)); } }
        fn parse_unary(&mut self) -> Result<Expr, String> { if self.match_tok(&Tok::Minus) { let e = self.parse_primary()?; Ok(Expr::BinOp(Box::new(Expr::Int(0)), BinOp::Sub, Box::new(e))) } else { self.parse_primary() } }
        fn parse_primary(&mut self) -> Result<Expr, String> { match self.peek().clone() { Tok::Int(n) => { self.advance(); Ok(Expr::Int(n)) } Tok::Float(f) => { self.advance(); Ok(Expr::Float(f)) } Tok::Str(s) => { self.advance(); Ok(Expr::Str(s)) } Tok::LParen => { self.advance(); let e = self.parse_expr()?; self.expect(&Tok::RParen)?; Ok(e) } Tok::LBrace => self.parse_block(), Tok::If => self.parse_if(), Tok::Match => self.parse_match(), Tok::Ident(name) => { self.advance(); if *self.peek() == Tok::LParen { self.advance(); let mut args = Vec::new(); if *self.peek() != Tok::RParen { args.push(self.parse_expr()?); while self.match_tok(&Tok::Comma) { args.push(self.parse_expr()?); } } self.expect(&Tok::RParen)?; Ok(Expr::Call(name, args)) } else { Ok(Expr::Var(name)) } } other => Err(format!("{}:{}: Unexpected token {:?}", self.line(), self.col(), other)), } }
        fn parse_if(&mut self) -> Result<Expr, String> { self.advance(); let cond = self.parse_expr()?; let then_e = self.parse_block()?; self.expect(&Tok::Else)?; let else_e = self.parse_block()?; Ok(Expr::If(Box::new(cond), Box::new(then_e), Box::new(else_e))) }
        fn parse_match(&mut self) -> Result<Expr, String> { self.advance(); let scrutinee = self.parse_expr()?; self.expect(&Tok::LBrace)?; let mut arms = Vec::new(); while *self.peek() != Tok::RBrace { let is_wildcard = *self.peek() == Tok::Ident("_".to_string()); let pattern = if is_wildcard { self.advance(); Expr::Int(0) } else { self.parse_expr()? }; self.expect(&Tok::FatArrow)?; let body = self.parse_expr()?; arms.push(MatchArm { pattern, body, is_wildcard }); if !self.match_tok(&Tok::Comma) { break; } } self.expect(&Tok::RBrace)?; Ok(Expr::Match(Box::new(scrutinee), arms)) }
    }
}

mod vm {
    use super::algebra::*; use std::collections::{BTreeSet, HashMap};
    #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum TokenKind { Unknown = 0, Letter = 1, Digit = 2, Delimiter = 3, Operator = 4, Keyword = 5, Identifier = 6, Literal = 7 }
    pub struct PhysicalState { pub omega_vector: CVector, pub representation: HashMap<GeneratorId, Matrix>, pub dagger_map: HashMap<GeneratorId, GeneratorId>, pub generator_names: HashMap<GeneratorId, String>, pub generator_kinds: HashMap<GeneratorId, TokenKind>, pub basis_dim: usize }
    impl PhysicalState {
        pub fn new(basis_dim: usize) -> Self { let mut ov = vec![Complex64::zero(); basis_dim]; ov[0] = Complex64::one(); Self { omega_vector: ov, representation: HashMap::new(), dagger_map: HashMap::new(), generator_names: HashMap::new(), generator_kinds: HashMap::new(), basis_dim } }
        pub fn open_extension(&mut self, increment: usize) {
            let old_basis = self.basis_dim;
            self.basis_dim += increment;
            println!("  [Substrate Physics] Open Extension triggered. Expanded algebraic degrees of freedom (basis): {} -> {}. Invariant 3D+1T inference channels preserved.", old_basis, self.basis_dim);
            self.omega_vector.resize(self.basis_dim, Complex64::zero());
            for mat in self.representation.values_mut() {
                for row in mat.iter_mut() { row.resize(self.basis_dim, Complex64::zero()); }
                for i in old_basis..self.basis_dim { let mut new_row = vec![Complex64::zero(); self.basis_dim]; new_row[i] = Complex64::one(); mat.push(new_row); }
            }
        }
        pub fn register_self_adjoint(&mut self, g: GeneratorId, mut matrix: Matrix) -> Result<(), String> {
            if matrix.len() != self.basis_dim {
                let old_basis = matrix.len();
                for row in matrix.iter_mut() { row.resize(self.basis_dim, Complex64::zero()); }
                for i in old_basis..self.basis_dim { let mut new_row = vec![Complex64::zero(); self.basis_dim]; new_row[i] = Complex64::one(); matrix.push(new_row); }
            }
            self.dagger_map.insert(g, g); self.representation.insert(g, matrix); Ok(())
        }
        fn represent_monomial(&self, m: &Monomial) -> Matrix { let mut result = identity(self.basis_dim); for g in &m.word { match self.representation.get(g) { Some(mat) => result = mat_mul(&result, mat), None => panic!("FATAL: Unknown generator {:?}", g), } } result }
        fn represent(&self, elem: &AlgebraElement) -> Matrix { let mut result = vec![vec![Complex64::zero(); self.basis_dim]; self.basis_dim]; for (m, c) in &elem.terms { let mm = self.represent_monomial(m); result = mat_add(&result, &mat_scale(&mm, *c)); } result }
        pub fn gns_vector(&self, elem: &AlgebraElement) -> CVector { let mat = self.represent(elem); mat_vec(&mat, &self.omega_vector) }
        pub fn restrict_state_to_constraint(&mut self, c_elem: &AlgebraElement) -> Result<(), String> { let mat = self.represent(c_elem); let ns = null_space(&mat); if ns.is_empty() { return Err("Constraint collapses state".to_string()); } let ortho_ns = gram_schmidt(&ns); let mut new_omega = vec![Complex64::zero(); self.basis_dim]; for e in &ortho_ns { let proj = inner(e, &self.omega_vector); for i in 0..self.basis_dim { new_omega[i] = new_omega[i] + proj * e[i]; } } let norm = norm_sq(&new_omega).sqrt(); if norm < 1e-12 { return Err("Constraint annihilates state".to_string()); } for i in 0..self.basis_dim { self.omega_vector[i] = new_omega[i] * (1.0 / norm); } Ok(()) }
        pub fn check_admissibility(&self, ideal: &ConstraintIdeal) -> AdmissibilityReport { let mut violations = Vec::new(); let mut max_v = 0.0f64; for (id, c) in &ideal.generators { let mat = self.represent(&c.element); let v = mat_vec(&mat, &self.omega_vector); let q = norm_sq(&v); if q > 1e-12 { violations.push(AdmissibilityViolation { constraint_id: *id, description: c.description.clone(), violation_magnitude: q }); max_v = max_v.max(q); } } AdmissibilityReport { is_admissible: violations.is_empty(), violations, max_violation: max_v } }
        pub fn correlation_kernel(&self, a: &AlgebraElement, b: &AlgebraElement) -> f64 { let va = self.gns_vector(a); let vb = self.gns_vector(b); let na = norm_sq(&va).sqrt(); let nb = norm_sq(&vb).sqrt(); if na < 1e-12 || nb < 1e-12 { return 0.0; } let overlap = inner(&va, &vb).modulus(); (overlap / (na * nb)).min(1.0) }
    }
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)] pub struct ConstraintId(pub u64);
    #[derive(Clone, Debug)] pub struct ConstraintOperator { pub id: ConstraintId, pub element: AlgebraElement, pub description: String }
    pub struct ConstraintIdeal { pub generators: HashMap<ConstraintId, ConstraintOperator> }
    impl ConstraintIdeal { pub fn new() -> Self { Self { generators: HashMap::new() } } pub fn add_constraint(&mut self, c: ConstraintOperator) { self.generators.insert(c.id, c); } }
    #[derive(Clone, Debug)] pub struct AdmissibilityReport { pub is_admissible: bool, pub violations: Vec<AdmissibilityViolation>, pub max_violation: f64 }
    #[derive(Clone, Debug)] pub struct AdmissibilityViolation { pub constraint_id: ConstraintId, pub description: String, pub violation_magnitude: f64 }
    pub struct GNSQuotient { pub classes: HashMap<Vec<u8>, QuotientClass>, pub threshold: f64 }
    #[derive(Clone, Debug)] pub struct QuotientClass { pub canonical_representative: AlgebraElement, pub gns_vector: CVector, pub signature: Vec<u8> }
    impl GNSQuotient { pub fn new(threshold: f64) -> Self { Self { classes: HashMap::new(), threshold } } fn compute_sig(v: &CVector) -> Vec<u8> { let mut s = Vec::with_capacity(v.len() * 16); for c in v { s.extend_from_slice(&(c.re / 1e-9).round().to_le_bytes()); s.extend_from_slice(&(c.im / 1e-9).round().to_le_bytes()); } s } pub fn resolve(&mut self, elem: &AlgebraElement, state: &PhysicalState) -> AlgebraElement { let gv = state.gns_vector(elem); let s = Self::compute_sig(&gv); if let Some(class) = self.classes.get(&s) { return class.canonical_representative.clone(); } for class in self.classes.values() { let mut diff = 0.0; for i in 0..gv.len().min(class.gns_vector.len()) { diff += (gv[i].re - class.gns_vector[i].re).powi(2) + (gv[i].im - class.gns_vector[i].im).powi(2); } if diff.sqrt() < self.threshold { return class.canonical_representative.clone(); } } let qc = QuotientClass { canonical_representative: elem.clone(), gns_vector: gv, signature: s.clone() }; self.classes.insert(s, qc); elem.clone() } }
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)] pub struct EventId(pub u64);
    #[derive(Clone, Debug)] pub struct CausalEvent { pub id: EventId, pub element: AlgebraElement, pub closure_set: BTreeSet<EventId>, pub prereqs: BTreeSet<EventId>, pub burden: f64, pub attentional_scale: f64, pub depth: usize, pub label: String }
    pub struct CausalGraph { pub events: HashMap<EventId, CausalEvent>, pub next_id: u64 }
    impl CausalGraph {
        pub fn new() -> Self { Self { events: HashMap::new(), next_id: 1 } }
        pub fn add_event(&mut self, elem: AlgebraElement, prereqs: BTreeSet<EventId>, burden: f64, label: String) -> EventId { let id = EventId(self.next_id); let mut closure = prereqs.clone(); for p in &prereqs { if let Some(e) = self.events.get(p) { closure.extend(e.closure_set.iter().cloned()); } } let depth = prereqs.iter().map(|p| self.events.get(p).map_or(0, |e| e.depth + 1)).max().unwrap_or(0); self.events.insert(id, CausalEvent { id, element: elem, closure_set: closure, prereqs, burden, attentional_scale: 1.0, depth, label }); self.next_id += 1; id }
        pub fn scale_memory_window(&mut self, lambda: f64) { for event in self.events.values_mut() { let clock_factor = 1.0 / (1.0 + lambda * event.burden); event.attentional_scale = (event.attentional_scale * clock_factor).max(1e-5); } }
        pub fn max_depth(&self) -> usize { self.events.values().map(|e| e.depth).max().unwrap_or(0) }
        pub fn max_chain_proper_time(&self, lambda: f64, epsilon: f64) -> f64 { if self.events.is_empty() { return 0.0; } let mut max_depth = 0; let mut deepest_event = None; for (id, e) in &self.events { if e.depth > max_depth { max_depth = e.depth; deepest_event = Some(*id); } } if deepest_event.is_none() { return 0.0; } let mut current_id = deepest_event.unwrap(); let mut tau = 0.0; loop { let event = match self.events.get(&current_id) { Some(e) => e, None => break }; tau += epsilon / (1.0 + lambda * event.burden); let mut next_event = None; let mut next_depth = -1; for &p in &event.prereqs { if let Some(pe) = self.events.get(&p) { if pe.depth as i32 > next_depth { next_depth = pe.depth as i32; next_event = Some(p); } } } match next_event { Some(p) => current_id = p, None => break } } tau }
        pub fn correlation_cluster(&self, eid: EventId, threshold: f64, state: &PhysicalState) -> Vec<EventId> { let target = match self.events.get(&eid) { Some(e) => &e.element, None => return Vec::new() }; let mut cluster = Vec::new(); for (&id, ev) in &self.events { if id == eid { continue; } let k = state.correlation_kernel(target, &ev.element); if k > threshold { cluster.push(id); } } cluster }
        pub fn causal_ancestors(&self, eid: EventId) -> Vec<EventId> { self.events.get(&eid).map(|e| e.closure_set.iter().cloned().collect()).unwrap_or_default() }
    }
    pub struct RCSVM { pub state: PhysicalState, pub ideal: ConstraintIdeal, pub quotient: GNSQuotient, pub graph: CausalGraph, pub next_gen: u64, pub next_constraint: u64, pub is_admissible: bool, pub lambda: f64, pub epsilon: f64, pub pathway_count: usize }
    impl RCSVM {
        pub fn new(initial_basis_dim: usize) -> Self { Self { state: PhysicalState::new(initial_basis_dim), ideal: ConstraintIdeal::new(), quotient: GNSQuotient::new(1e-9), graph: CausalGraph::new(), next_gen: 1, next_constraint: 0, is_admissible: true, lambda: 0.1, epsilon: 1.0, pathway_count: 0 } }
        pub fn register_generator(&mut self) -> Result<GeneratorId, String> {
            // Saturate when pathways fill the basis (leave 1 slot for vacuum)
            if self.pathway_count >= self.state.basis_dim {
                self.state.open_extension(1);
            }
            let g = GeneratorId(self.next_gen); self.next_gen += 1;
            self.state.register_self_adjoint(g, generator_matrix(self.state.basis_dim, g.0))?;
            self.pathway_count += 1;
            Ok(g)
        }
        pub fn register_generator_with_kind(&mut self, name: &str, kind: TokenKind) -> Result<GeneratorId, String> {
            let g = self.register_generator()?;
            self.state.generator_names.insert(g, name.to_string());
            self.state.generator_kinds.insert(g, kind);
            Ok(g)
        }
        pub fn register_generator_with_value(&mut self, value: f64) -> Result<GeneratorId, String> {
            if self.pathway_count >= self.state.basis_dim {
                self.state.open_extension(1);
            }
            let g = GeneratorId(self.next_gen); self.next_gen += 1;
            let mat = mat_scale(&identity(self.state.basis_dim), Complex64::real(value));
            self.state.register_self_adjoint(g, mat)?;
            self.pathway_count += 1;
            Ok(g)
        }
        pub fn register_generator_with_matrix(&mut self, matrix: Matrix) -> Result<GeneratorId, String> {
            if self.pathway_count >= self.state.basis_dim {
                self.state.open_extension(1);
            }
            let g = GeneratorId(self.next_gen); self.next_gen += 1;
            self.state.register_self_adjoint(g, matrix)?;
            self.pathway_count += 1;
            Ok(g)
        }
        pub fn add_constraint(&mut self, elem: AlgebraElement, desc: String, restrict_state: bool) -> Result<ConstraintId, String> { let cid = ConstraintId(self.next_constraint); self.next_constraint += 1; self.ideal.add_constraint(ConstraintOperator { id: cid, element: elem.clone(), description: desc }); if restrict_state { self.state.restrict_state_to_constraint(&elem)?; } Ok(cid) }
        pub fn execute(&mut self, elem: AlgebraElement, prereqs: BTreeSet<EventId>, label: String) -> Result<EventId, String> { let canonical_elem = self.quotient.resolve(&elem, &self.state); let report = self.state.check_admissibility(&self.ideal); self.is_admissible = report.is_admissible; let mut inherited_burden = 0.0; for &p in &prereqs { if let Some(e) = self.graph.events.get(&p) { inherited_burden += e.burden * self.lambda * 0.5; } } let physical_burden = 1.0 + report.max_violation + inherited_burden; let eid = self.graph.add_event(canonical_elem, prereqs, physical_burden, label); self.graph.scale_memory_window(self.lambda); if !self.is_admissible { return Err(format!("Constraint violation. Max tension: {}", report.max_violation)); } Ok(eid) }
        pub fn compression_ratio(&self) -> f64 { let expressions_ingested = self.graph.events.len() as f64; let active_sectors = self.quotient.classes.len() as f64; if active_sectors == 0.0 { 1.0 } else { expressions_ingested / active_sectors } }
        pub fn get_event_tokens(&self, eid: EventId) -> Option<Vec<GeneratorId>> { self.graph.events.get(&eid).and_then(|e| e.element.extract_token_sequence()) }
    }
}

mod compiler {
    use super::ast::*; use super::vm::*; use super::algebra::*; use std::collections::{BTreeSet, HashMap, HashSet};
    #[derive(Clone)] pub struct EventTemplate { pub params: Vec<(String, String)>, pub requires: Vec<Expr>, pub body: Expr }
    #[derive(Clone)] pub struct ConstraintTemplate { pub params: Vec<(String, String)>, pub condition: Expr }
    pub struct Compiler { pub generators: HashMap<String, GeneratorId>, pub concepts: HashMap<String, GeneratorId>, pub events: HashMap<String, EventTemplate>, pub constraints: HashMap<String, ConstraintTemplate>, pub let_bindings: HashMap<String, AlgebraElement>, pub let_strings: HashMap<String, String>, pub imported_files: HashSet<String> }
    impl Compiler {
        pub fn new() -> Self { Self { generators: HashMap::new(), concepts: HashMap::new(), events: HashMap::new(), constraints: HashMap::new(), let_bindings: HashMap::new(), let_strings: HashMap::new(), imported_files: HashSet::new() } }
        pub fn ingest_text(&mut self, text: &str, vm: &mut RCSVM) -> Result<EventId, String> { let mut remaining = text; let mut elem = AlgebraElement::unit(); while !remaining.is_empty() { let mut best_match = ""; for phrase in self.concepts.keys() { if remaining.starts_with(phrase) && phrase.len() > best_match.len() { best_match = phrase; } } if !best_match.is_empty() { let &g = self.concepts.get(best_match).unwrap(); elem = elem.mul(&AlgebraElement::generator(g)); remaining = &remaining[best_match.len()..]; } else { if remaining.starts_with(char::is_whitespace) { remaining = &remaining[1..]; continue; } let mut end = 0; for (i, c) in remaining.char_indices() { if c.is_whitespace() { break; } end = i + c.len_utf8(); } if end == 0 { break; } let word = &remaining[..end]; let clean_word = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase(); if !clean_word.is_empty() { let g = if let Some(&g) = self.generators.get(&clean_word) { g } else { let new_g = vm.register_generator_with_kind(&clean_word, TokenKind::Identifier)?; self.generators.insert(clean_word.clone(), new_g); new_g }; elem = elem.mul(&AlgebraElement::generator(g)); } remaining = &remaining[end..]; } } let eid = vm.execute(elem, BTreeSet::new(), format!("ingest:{}", text))?; Ok(eid) }
        pub fn lex_source(&mut self, source: &str, vm: &mut RCSVM) -> Result<f64, String> { let mut elem = AlgebraElement::unit(); let mut chars = source.chars().peekable(); while let Some(&c) = chars.peek() { if c.is_whitespace() { chars.next(); continue; } if c.is_alphabetic() { let mut word = String::new(); while let Some(&c) = chars.peek() { if c.is_alphanumeric() || c == '_' { word.push(c); chars.next(); } else { break; } } let kind = if is_keyword(&word) { TokenKind::Keyword } else { TokenKind::Identifier }; let g = if let Some(&g) = self.generators.get(&word) { g } else { let new_g = vm.register_generator_with_kind(&word, kind)?; self.generators.insert(word.clone(), new_g); new_g }; elem = elem.mul(&AlgebraElement::generator(g)); } else if c.is_numeric() { let mut num = String::new(); while let Some(&c) = chars.peek() { if c.is_numeric() { num.push(c); chars.next(); } else { break; } } let g = vm.register_generator_with_kind(&num, TokenKind::Literal)?; elem = elem.mul(&AlgebraElement::generator(g)); } else { let mut sym = String::new(); sym.push(c); chars.next(); if let Some(&c2) = chars.peek() { let two_char = format!("{}{}", c, c2); if ["==", "=>", "->", "!=", ">=", "<="].contains(&two_char.as_str()) { sym.push(c2); chars.next(); } } let kind = if "+-*/=".contains(&sym) { TokenKind::Operator } else { TokenKind::Delimiter }; let g = vm.register_generator_with_kind(&sym, kind)?; elem = elem.mul(&AlgebraElement::generator(g)); } } let eid = vm.execute(elem, BTreeSet::new(), format!("lex:{}", source))?; Ok(eid.0 as f64) }
        pub fn compile(&mut self, program: &Program, vm: &mut RCSVM) -> Result<(), String> {
            for decl in &program.decls { match decl {
                Decl::Import(path) => {
                    if self.imported_files.contains(path) { println!("  [Module System] Already imported '{}', skipping.", path); continue; }
                    self.imported_files.insert(path.clone());
                    println!("  [Module System] Importing '{}'...", path);
                    let content = std::fs::read_to_string(path).map_err(|e| format!("Cannot import '{}': {}", path, e))?;
                    let tokens = crate::lexer::Lexer::new(&content).tokenize().map_err(|e| format!("Lex error in '{}': {}", path, e))?;
                    let program = crate::parser::Parser::new(tokens).parse_program().map_err(|e| format!("Parse error in '{}': {}", path, e))?;
                    let mut module_compiler = Compiler::new();
                    module_compiler.imported_files = self.imported_files.clone();
                    module_compiler.compile(&program, vm)?;
                    for (name, tmpl) in &module_compiler.events { self.events.insert(name.clone(), tmpl.clone()); }
                    for (name, tmpl) in &module_compiler.constraints { self.constraints.insert(name.clone(), tmpl.clone()); }
                    for (name, &g) in &module_compiler.concepts { self.concepts.insert(name.clone(), g); }
                    for (name, &g) in &module_compiler.generators { self.generators.insert(name.clone(), g); }
                    for (name, elem) in &module_compiler.let_bindings { self.let_bindings.insert(name.clone(), elem.clone()); }
                    for (name, s) in &module_compiler.let_strings { self.let_strings.insert(name.clone(), s.clone()); }
                    self.imported_files = module_compiler.imported_files;
                    println!("  [Module System] Imported '{}' successfully.", path);
                }
                Decl::Generator { name, ty: _, init } => { let g = if let Some(init_expr) = init { let (val, _) = self.eval_expr(init_expr, vm, &HashMap::new(), &self.let_strings.clone())?; if let Some(scalar) = val.as_scalar() { vm.register_generator_with_value(scalar)? } else { vm.register_generator_with_kind(name, TokenKind::Identifier)? } } else { vm.register_generator_with_kind(name, TokenKind::Identifier)? }; self.generators.insert(name.clone(), g); }
                Decl::Concept { name: _, tokens } => { if tokens.is_empty() { continue; } let canonical_phrase = &tokens[0]; let canonical_g = if let Some(&g) = self.generators.get(canonical_phrase) { g } else { let g = vm.register_generator_with_kind(canonical_phrase, TokenKind::Identifier)?; self.generators.insert(canonical_phrase.clone(), g); g }; self.concepts.insert(canonical_phrase.clone(), canonical_g); let canonical_mat = vm.state.representation.get(&canonical_g).cloned().unwrap(); for phrase in tokens.iter().skip(1) { let new_g = vm.register_generator_with_matrix(canonical_mat.clone())?; vm.state.generator_names.insert(new_g, phrase.clone()); vm.state.generator_kinds.insert(new_g, TokenKind::Identifier); self.generators.insert(phrase.clone(), new_g); self.concepts.insert(phrase.clone(), new_g); let constraint_elem = AlgebraElement::generator(new_g).sub(&AlgebraElement::generator(canonical_g)); vm.add_constraint(constraint_elem, format!("ConceptMerge: {}", phrase), false)?; } }
                Decl::Constraint { name, params, condition } => { self.constraints.insert(name.clone(), ConstraintTemplate { params: params.clone(), condition: condition.clone() }); }
                Decl::Event { name, params, requires, body, .. } => { self.events.insert(name.clone(), EventTemplate { params: params.clone(), requires: requires.clone(), body: body.clone() }); }
            }}
            for stmt in &program.stmts { match stmt {
                Stmt::Ingest(text) => { let _ = self.ingest_text(text, vm)?; }
                Stmt::Let { name, value } => { if let Expr::Str(s) = value { self.let_strings.insert(name.clone(), s.clone()); } else { let (elem, prereqs) = self.eval_expr(value, vm, &HashMap::new(), &self.let_strings.clone())?; self.let_bindings.insert(name.clone(), elem.clone()); let g = if let Some(scalar) = elem.as_scalar() { vm.register_generator_with_value(scalar)? } else { vm.register_generator_with_kind(name, TokenKind::Identifier)? }; self.generators.insert(name.clone(), g); vm.execute(elem, prereqs, format!("let {}", name))?; } }
                Stmt::Print(expr) => { let (elem, _) = self.eval_expr(expr, vm, &HashMap::new(), &self.let_strings.clone())?; if let Some(scalar) = elem.as_scalar() { println!("Print: {}", scalar); } else { let val = vm.state.gns_vector(&elem); println!("Print profile [Vector Size: {} -> Norm: {:.4}]", val.len(), norm_sq(&val)); } }
                Stmt::Expr(expr) => { let (elem, prereqs) = self.eval_expr(expr, vm, &HashMap::new(), &self.let_strings.clone())?; vm.execute(elem, prereqs, "expr_eval".to_string())?; }
            }}
            Ok(())
        }
        fn eval_expr(&mut self, expr: &Expr, vm: &mut RCSVM, scope: &HashMap<String, AlgebraElement>, strings: &HashMap<String, String>) -> Result<(AlgebraElement, BTreeSet<EventId>), String> {
            match expr {
                Expr::Int(n) => Ok((AlgebraElement::scalar(*n as f64), BTreeSet::new())),
                Expr::Float(f) => Ok((AlgebraElement::scalar(*f), BTreeSet::new())),
                Expr::Var(name) => {
                    if let Some(elem) = scope.get(name) { Ok((elem.clone(), BTreeSet::new())) }
                    else if let Some(elem) = self.let_bindings.get(name) { Ok((elem.clone(), BTreeSet::new())) }
                    else if let Some(&g) = self.generators.get(name) {
                        if let Some(mat) = vm.state.representation.get(&g) {
                            if mat.len() == vm.state.basis_dim && !mat.is_empty() && mat[0].len() == vm.state.basis_dim {
                                let mut is_scalar_mat = true; let val = mat[0][0].re;
                                if mat[0][0].im.abs() > 1e-10 { is_scalar_mat = false; }
                                else { for i in 0..vm.state.basis_dim { for j in 0..vm.state.basis_dim { if i == j { if (mat[i][j].re - val).abs() > 1e-10 || mat[i][j].im.abs() > 1e-10 { is_scalar_mat = false; break; } } else { if mat[i][j].norm_sq() > 1e-12 { is_scalar_mat = false; break; } } } if !is_scalar_mat { break; } } }
                                if is_scalar_mat { return Ok((AlgebraElement::scalar(val), BTreeSet::new())); }
                            }
                        }
                        Ok((AlgebraElement::generator(g), BTreeSet::new()))
                    } else if strings.contains_key(name) { Err(format!("Variable {} is a string, cannot be used in algebraic context", name)) }
                    else { Err(format!("Reference error: {}", name)) }
                }
                Expr::BinOp(l, op, r) => { let (le, mut prereqs) = self.eval_expr(l, vm, scope, strings)?; let (re, r_prereqs) = self.eval_expr(r, vm, scope, strings)?; prereqs.extend(r_prereqs); match op { BinOp::Add => Ok((le.add(&re), prereqs)), BinOp::Sub => Ok((le.sub(&re), prereqs)), BinOp::Mul => Ok((le.mul(&re), prereqs)), BinOp::Div => { let ls = le.as_scalar().ok_or("Division requires scalar")?; let rs = re.as_scalar().ok_or("Division requires scalar")?; if rs.abs() < 1e-15 { return Err("Division by zero".to_string()); } Ok((AlgebraElement::scalar(ls / rs), prereqs)) } BinOp::Eq | BinOp::NotEq | BinOp::Gte | BinOp::Lte | BinOp::Gt | BinOp::Lt => { let ls = le.as_scalar().ok_or("Comparison requires scalar")?; let rs = re.as_scalar().ok_or("Comparison requires scalar")?; let result = match op { BinOp::Eq => (ls - rs).abs() < 1e-10, BinOp::NotEq => (ls - rs).abs() >= 1e-10, BinOp::Gte => ls >= rs, BinOp::Lte => ls <= rs, BinOp::Gt => ls > rs, BinOp::Lt => ls < rs, _ => false, }; Ok((AlgebraElement::scalar(if result { 1.0 } else { 0.0 }), prereqs)) } } }
                Expr::If(cond, then_e, else_e) => { let (cond_val, mut prereqs) = self.eval_expr(cond, vm, scope, strings)?; let cond_scalar = cond_val.as_scalar().ok_or("If condition must be scalar")?; if cond_scalar.abs() > 0.5 { let (val, p) = self.eval_expr(then_e, vm, scope, strings)?; prereqs.extend(p); Ok((val, prereqs)) } else { let (val, p) = self.eval_expr(else_e, vm, scope, strings)?; prereqs.extend(p); Ok((val, prereqs)) } }
                Expr::Call(name, args) => {
                    // Intercept 'apply' FIRST to prevent generic argument evaluation from crashing on string variables
                    if name == "apply" {
                        if args.len() != 2 { return Err("apply expects 2 args: (event_name, arg_id)".to_string()); }
                        
                        let func_name = match &args[0] {
                            Expr::Str(s) => s.clone(),
                            Expr::Var(n) => strings.get(n).cloned()
                                .or_else(|| self.let_strings.get(n).cloned())
                                .ok_or(format!("Cannot resolve string variable '{}' for apply", n))?,
                            _ => return Err("apply requires a string event name or string variable".to_string()),
                        };
                        
                        let (val, _) = self.eval_expr(&args[1], vm, scope, strings)?;
                        
                        let template = self.events.get(&func_name).ok_or(format!("Unknown event: {}", func_name))?.clone();
                        let mut local_scope = scope.clone();
                        if template.params.len() >= 1 {
                            local_scope.insert(template.params[0].0.clone(), val);
                        }
                        
                        let (result, body_prereqs) = self.eval_expr(&template.body, vm, &local_scope, strings)?;
                        let mut prereqs = BTreeSet::new();
                        prereqs.extend(body_prereqs);
                        
                        let eid = vm.execute(result.clone(), prereqs, format!("apply:{}", func_name))?;
                        return Ok((AlgebraElement::scalar(eid.0 as f64), BTreeSet::new()));
                    }
                
                    match name.as_str() {
                        "print_str" => {
                            if args.len() != 1 { return Err("print_str expects 1 string".to_string()); }
                            let s = match &args[0] {
                                Expr::Str(s) => s.clone(),
                                Expr::Var(n) => strings.get(n).cloned().or_else(|| self.let_strings.get(n).cloned()).ok_or("String not found")?,
                                _ => return Err("print_str requires a string".to_string()),
                            };
                            println!("{}", s);
                            return Ok((AlgebraElement::scalar(1.0), BTreeSet::new()));
                        }
                        "lex" => { let s = match &args[0] { Expr::Str(s) => s.clone(), Expr::Var(n) => strings.get(n).cloned().ok_or(format!("String variable {} not found", n))?, _ => return Err("lex requires a string or string variable".to_string()), }; let eid = self.lex_source(&s, vm)?; return Ok((AlgebraElement::scalar(eid), BTreeSet::new())); }
                        "lex_file" => { let path = match &args[0] { Expr::Str(s) => s.clone(), Expr::Var(n) => strings.get(n).cloned().ok_or(format!("String variable {} not found", n))?, _ => return Err("lex_file requires a string or string variable".to_string()), }; let content = std::fs::read_to_string(&path).map_err(|e| format!("File I/O Error '{}': {}", path, e))?; let eid = self.lex_source(&content, vm)?; return Ok((AlgebraElement::scalar(eid), BTreeSet::new())); }
                        "lex_file_lines" => {
                            if args.len() != 1 { return Err("lex_file_lines expects 1 string path".to_string()); }
                            let path = match &args[0] { Expr::Str(s) => s.clone(), Expr::Var(n) => strings.get(n).cloned().ok_or(format!("String variable {} not found", n))?, _ => return Err("lex_file_lines requires a string path".to_string()) };
                            let content = std::fs::read_to_string(&path).map_err(|e| format!("File I/O Error '{}': {}", path, e))?;
                            let mut prev_id = 0.0;
                            let lines: Vec<&str> = content.lines().collect();
                            for line in lines.into_iter().rev() {
                                if !line.is_empty() {
                                    let mut elem = AlgebraElement::unit();
                                    let mut chars = line.chars().peekable();
                                    while let Some(&c) = chars.peek() {
                                        if c.is_whitespace() { chars.next(); continue; }
                                        if c.is_alphabetic() {
                                            let mut word = String::new();
                                            while let Some(&c) = chars.peek() { if c.is_alphanumeric() || c == '_' { word.push(c); chars.next(); } else { break; } }
                                            let kind = if is_keyword(&word) { TokenKind::Keyword } else { TokenKind::Identifier };
                                            let g = if let Some(&g) = self.generators.get(&word) { g } else { let new_g = vm.register_generator_with_kind(&word, kind)?; self.generators.insert(word.clone(), new_g); new_g };
                                            elem = elem.mul(&AlgebraElement::generator(g));
                                        } else if c.is_numeric() {
                                            let mut num = String::new();
                                            while let Some(&c) = chars.peek() { if c.is_numeric() { num.push(c); chars.next(); } else { break; } }
                                            let g = vm.register_generator_with_kind(&num, TokenKind::Literal)?;
                                            elem = elem.mul(&AlgebraElement::generator(g));
                                        } else {
                                            let mut sym = String::new();
                                            sym.push(c);
                                            chars.next();
                                            if let Some(&c2) = chars.peek() { let two_char = format!("{}{}", c, c2); if ["==", "=>", "->", "!=", ">=", "<="].contains(&two_char.as_str()) { sym.push(c2); chars.next(); } }
                                            let kind = if "+-*/=".contains(&sym) { TokenKind::Operator } else { TokenKind::Delimiter };
                                            let g = vm.register_generator_with_kind(&sym, kind)?;
                                            elem = elem.mul(&AlgebraElement::generator(g));
                                        }
                                    }
                                    let mut prereqs = BTreeSet::new();
                                    if prev_id > 0.0 { prereqs.insert(EventId(prev_id as u64)); }
                                    let new_eid = vm.execute(elem, prereqs, format!("stream_line:{}", line))?;
                                    prev_id = new_eid.0 as f64;
                                }
                            }
                            println!("  [Substrate Stream] Ingested {} as causal list (head: {})", path, prev_id);
                            return Ok((AlgebraElement::scalar(prev_id), BTreeSet::new()));
                        }
                        "token_count" => { let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let eid = e.as_scalar().ok_or("Expected event id")? as u64; let count = vm.get_event_tokens(EventId(eid)).map(|t| t.len()).unwrap_or(0); return Ok((AlgebraElement::scalar(count as f64), BTreeSet::new())); }
                        "token_kind" => { let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let (i, _) = self.eval_expr(&args[1], vm, scope, strings)?; let eid = e.as_scalar().ok_or("Expected event id")? as u64; let idx = i.as_scalar().ok_or("Expected index")? as usize; let kind = vm.get_event_tokens(EventId(eid)).and_then(|t| t.get(idx).copied()).and_then(|g| vm.state.generator_kinds.get(&g).copied()).unwrap_or(TokenKind::Unknown); return Ok((AlgebraElement::scalar(kind as u8 as f64), BTreeSet::new())); }
                        "token_is" => { let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let (i, _) = self.eval_expr(&args[1], vm, scope, strings)?; let text = if let Some(Expr::Str(s)) = args.get(2) { s.clone() } else { return Err("token_is requires string".to_string()); }; let eid = e.as_scalar().ok_or("Expected event id")? as u64; let idx = i.as_scalar().ok_or("Expected index")? as usize; let is_match = vm.get_event_tokens(EventId(eid)).and_then(|t| t.get(idx).copied()).and_then(|g| vm.state.generator_names.get(&g)).map(|n| n == &text).unwrap_or(false); return Ok((AlgebraElement::scalar(if is_match { 1.0 } else { 0.0 }), BTreeSet::new())); }
                        "bind_token_as_generator" => { if args.len() != 2 { return Err("bind_token_as_generator expects 2 args".to_string()); } let (ev_val, _) = self.eval_expr(&args[0], vm, scope, strings)?; let (idx_val, _) = self.eval_expr(&args[1], vm, scope, strings)?; let ev_id = ev_val.as_scalar().ok_or("ev_id must be scalar")? as u64; let token_idx = idx_val.as_scalar().ok_or("token_index must be scalar")? as usize; let event = vm.graph.events.get(&EventId(ev_id)).ok_or(format!("Event {} not found", ev_id))?; let tokens = event.element.extract_token_sequence().ok_or("Failed to extract token sequence")?; if token_idx >= tokens.len() { return Err(format!("Token index {} out of bounds", token_idx)); } let target_gen_id = tokens[token_idx]; return Ok((AlgebraElement::scalar(target_gen_id.0 as f64), BTreeSet::new())); }
                        "add_constraint" => { if args.len() != 1 { return Err("add_constraint expects 1 arg".to_string()); } let (elem, _) = self.eval_expr(&args[0], vm, scope, strings)?; let cid = vm.add_constraint(elem, "Dynamic RCL Constraint".to_string(), true)?; return Ok((AlgebraElement::scalar(cid.0 as f64), BTreeSet::new())); }
                        "execute" => { if args.len() != 1 { return Err("execute expects 1 arg".to_string()); } let (elem, body_prereqs) = self.eval_expr(&args[0], vm, scope, strings)?; let eid = vm.execute(elem, body_prereqs, "dynamic_execute".to_string())?; return Ok((AlgebraElement::scalar(eid.0 as f64), BTreeSet::new())); }
                        "execute_after" => { if args.len() != 2 { return Err("execute_after expects 2 args".to_string()); } let (elem, mut body_prereqs) = self.eval_expr(&args[0], vm, scope, strings)?; let (prereq_val, _) = self.eval_expr(&args[1], vm, scope, strings)?; let pid = prereq_val.as_scalar().ok_or("Prerequisite must be scalar")? as u64; if pid > 0 { body_prereqs.insert(EventId(pid)); } let eid = vm.execute(elem, body_prereqs, "dynamic_sequential_execute".to_string())?; return Ok((AlgebraElement::scalar(eid.0 as f64), BTreeSet::new())); }
                        "event_element" => { if args.len() != 1 { return Err("event_element expects 1 arg".to_string()); } let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let eid_val = e.as_scalar().ok_or("event_element expects scalar event ID")?; let eid = EventId(eid_val as u64); let event = vm.graph.events.get(&eid).ok_or(format!("Event {} not found", eid.0))?; return Ok((event.element.clone(), BTreeSet::new())); }
                        "head" => { if args.len() != 1 { return Err("head expects 1 event id".to_string()); } let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let eid = e.as_scalar().ok_or("Expected event id")? as u64; if eid == 0 { return Ok((AlgebraElement::zero(), BTreeSet::new())); } let event = vm.graph.events.get(&EventId(eid)).ok_or("Event not found")?; return Ok((event.element.clone(), BTreeSet::new())); }
                        "tail" | "first_prereq" => { if args.len() != 1 { return Err("tail expects 1 event id".to_string()); } let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?; let eid = e.as_scalar().ok_or("Expected event id")? as u64; if eid == 0 { return Ok((AlgebraElement::scalar(0.0), BTreeSet::new())); } let event = vm.graph.events.get(&EventId(eid)).ok_or("Event not found")?; if let Some(&p) = event.prereqs.iter().next() { return Ok((AlgebraElement::scalar(p.0 as f64), BTreeSet::new())); } return Ok((AlgebraElement::scalar(0.0), BTreeSet::new())); }
                        // --- Native Standard Library: Causal List ---
                        "cons" => {
                            if args.len() != 2 { return Err("cons expects 2 args: (head_val, tail_id)".to_string()); }
                            let (h_elem, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let (t_elem, _) = self.eval_expr(&args[1], vm, scope, strings)?;
                            let tail_id = t_elem.as_scalar().ok_or("tail_id must be a scalar event id")? as u64;
                            
                            let mut prereqs = BTreeSet::new();
                            if tail_id > 0 { prereqs.insert(EventId(tail_id)); }
                            
                            let new_eid = vm.execute(h_elem, prereqs, "cons".to_string())?;
                            return Ok((AlgebraElement::scalar(new_eid.0 as f64), BTreeSet::new()));
                        }
                        
                        "length" => {
                            if args.len() != 1 { return Err("length expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let mut eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            
                            let mut count = 0.0;
                            while eid != 0 {
                                count += 1.0;
                                let event = match vm.graph.events.get(&EventId(eid)) {
                                    Some(ev) => ev,
                                    None => break,
                                };
                                eid = if let Some(&p) = event.prereqs.iter().next() { p.0 } else { 0 };
                            }
                            return Ok((AlgebraElement::scalar(count), BTreeSet::new()));
                        }
                        
                        "sum" => {
                            if args.len() != 1 { return Err("sum expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let mut eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            
                            let mut total = 0.0;
                            while eid != 0 {
                                let event = match vm.graph.events.get(&EventId(eid)) {
                                    Some(ev) => ev,
                                    None => break,
                                };
                                if let Some(val) = event.element.as_scalar() {
                                    total += val;
                                }
                                eid = if let Some(&p) = event.prereqs.iter().next() { p.0 } else { 0 };
                            }
                            return Ok((AlgebraElement::scalar(total), BTreeSet::new()));
                        }
                        "correlate" => { if args.len() != 2 { return Err("correlate expects 2 args".to_string()); } let (le, _) = self.eval_expr(&args[0], vm, scope, strings)?; let (re, _) = self.eval_expr(&args[1], vm, scope, strings)?; let elem_a = if let Some(scalar) = le.as_scalar() { let eid = EventId(scalar as u64); vm.graph.events.get(&eid).map(|ev| ev.element.clone()).unwrap_or(le) } else { le }; let elem_b = if let Some(scalar) = re.as_scalar() { let eid = EventId(scalar as u64); vm.graph.events.get(&eid).map(|ev| ev.element.clone()).unwrap_or(re) } else { re }; let k = vm.state.correlation_kernel(&elem_a, &elem_b); return Ok((AlgebraElement::scalar(k), BTreeSet::new())); }
                        "born_weight" => {
                            if args.len() != 2 { return Err("born_weight expects 2 args: (obs_id, proto_id)".to_string()); }
                            let (le, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let (re, _) = self.eval_expr(&args[1], vm, scope, strings)?;
                            
                            let elem_a = if let Some(scalar) = le.as_scalar() {
                                let eid = EventId(scalar as u64);
                                vm.graph.events.get(&eid).map(|ev| ev.element.clone()).unwrap_or(le)
                            } else { le };
                            
                            let elem_b = if let Some(scalar) = re.as_scalar() {
                                let eid = EventId(scalar as u64);
                                vm.graph.events.get(&eid).map(|ev| ev.element.clone()).unwrap_or(re)
                            } else { re };
                            
                            let va = vm.state.gns_vector(&elem_a);
                            let vb = vm.state.gns_vector(&elem_b);
                            let na = norm_sq(&va).sqrt();
                            let nb = norm_sq(&vb).sqrt();
                            
                            if na < 1e-12 || nb < 1e-12 {
                                return Ok((AlgebraElement::scalar(0.0), BTreeSet::new()));
                            }
                            
                            let overlap = inner(&vb, &va);
                            // Normalized Born Rule: |<b|a>|^2 / (|a|^2 * |b|^2)
                            let weight = (overlap.modulus() / (na * nb)).powi(2);
                            
                            return Ok((AlgebraElement::scalar(weight), BTreeSet::new()));
                        }
                        "check_horizon" => {
                            // Returns the exterior accessibility ratio ε_R = 1/(1+λB)
                            // When ε_R drops below a threshold, the event is "behind a horizon"
                            if args.len() != 1 { return Err("check_horizon expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            if eid == 0 { return Ok((AlgebraElement::scalar(1.0), BTreeSet::new())); }
                            
                            let event = vm.graph.events.get(&EventId(eid)).ok_or("Event not found")?;
                            let epsilon_r = event.attentional_scale;
                            return Ok((AlgebraElement::scalar(epsilon_r), BTreeSet::new()));
                        }
                        
                        "boundary_record" => {
                            // Extracts the boundary record of an unreconciled region.
                            // Returns the number of causal links that still cross the horizon.
                            // If this is > 0, the interior is recoverable (holographic principle).
                            if args.len() != 1 { return Err("boundary_record expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            if eid == 0 { return Ok((AlgebraElement::scalar(0.0), BTreeSet::new())); }
                            
                            let event = vm.graph.events.get(&EventId(eid)).ok_or("Event not found")?;
                            
                            // Count how many descendants reference this event (links crossing outward)
                            let mut boundary_links = 0.0;
                            for (&other_id, other_ev) in &vm.graph.events {
                                if other_id.0 != eid && other_ev.closure_set.contains(&EventId(eid)) {
                                    // Check if the descendant is still accessible (not also behind horizon)
                                    if other_ev.attentional_scale > 0.01 {
                                        boundary_links += 1.0;
                                    }
                                }
                            }
                            
                            return Ok((AlgebraElement::scalar(boundary_links), BTreeSet::new()));
                        }

                        "region_entropy" => {
                            // Computes the coarse-grained entropy of a region.
                            if args.len() != 1 { return Err("region_entropy expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            if eid == 0 { return Ok((AlgebraElement::scalar(0.0), BTreeSet::new())); }
                            
                            let event = vm.graph.events.get(&EventId(eid)).ok_or("Event not found")?;
                            
                            // Explicitly type as f64 so .ln() can resolve
                            let mut boundary_links: f64 = 0.0;
                            for (&other_id, other_ev) in &vm.graph.events {
                                if other_id.0 != eid && other_ev.closure_set.contains(&EventId(eid)) {
                                    if other_ev.attentional_scale > 0.01 {
                                        boundary_links += 1.0;
                                    }
                                }
                            }
                            
                            let entropy = (1.0f64 + boundary_links).ln();
                            return Ok((AlgebraElement::scalar(entropy), BTreeSet::new()));
                        }
                        
                        "emit_hawking" => {
                            // Finds the lowest-burden event in the causal graph and returns it.
                            // This is the "simplest admissible carrier sector" — the system relaxes
                            // by shedding its lightest structural load first.
                            if args.len() != 0 { return Err("emit_hawking expects 0 args".to_string()); }
                            
                            let mut min_burden = f64::MAX;
                            let mut min_eid = 0.0;
                            
                            for (&id, ev) in &vm.graph.events {
                                // Only consider events with burden > 0 (not freshly created)
                                if ev.burden > 0.0 && ev.burden < min_burden {
                                    min_burden = ev.burden;
                                    min_eid = id.0 as f64;
                                }
                            }
                            
                            if min_eid > 0.0 {
                                println!("  [Black Hole Physics] Hawking emission: Event {} (burden: {:.4})", min_eid as u64, min_burden);
                            }
                            
                            return Ok((AlgebraElement::scalar(min_eid), BTreeSet::new()));
                        }
                        
                        "isolate_sector" => {
                            // Forces an event's attentional scale to near-zero, simulating
                            // extreme burden isolation without destroying the algebraic state.
                            // The event becomes an unreconciled region — information preserved,
                            // but exterior accessibility collapses.
                            if args.len() != 1 { return Err("isolate_sector expects 1 event id".to_string()); }
                            let (e, _) = self.eval_expr(&args[0], vm, scope, strings)?;
                            let eid = e.as_scalar().ok_or("Expected event id")? as u64;
                            if eid == 0 { return Ok((AlgebraElement::scalar(0.0), BTreeSet::new())); }
                            
                            let event = vm.graph.events.get_mut(&EventId(eid)).ok_or("Event not found")?;
                            
                            // Push burden to extreme, collapse attentional scale
                            event.burden = 1000.0;
                            event.attentional_scale = 1.0 / (1.0 + vm.lambda * 1000.0);
                            
                            println!("  [Black Hole Physics] Event {} isolated behind horizon. ε_R = {:.6}", eid, event.attentional_scale);
                            return Ok((AlgebraElement::scalar(1.0), BTreeSet::new()));
                        }
                        
                        "project_sector" => { if args.len() != 2 { return Err("project_sector expects 2 args".to_string()); } let (elem, _) = self.eval_expr(&args[0], vm, scope, strings)?; let label = match &args[1] { Expr::Str(s) => s.clone(), _ => return Err("project_sector label must be string".to_string()), }; vm.state.restrict_state_to_constraint(&elem)?; let cid = vm.add_constraint(elem, format!("Sector: {}", label), false)?; return Ok((AlgebraElement::scalar(cid.0 as f64), BTreeSet::new())); }
                        "age_events" => { if args.len() != 1 { return Err("age_events expects 1 scalar argument".to_string()); } let (dt_elem, _) = self.eval_expr(&args[0], vm, scope, strings)?; let dt = dt_elem.as_scalar().ok_or("delta_t must be a scalar float")?; for (_, event) in vm.graph.events.iter_mut() { event.burden += dt; event.attentional_scale = 1.0 / (1.0 + vm.lambda * event.burden); } println!("  [Substrate Physics] Proper time advanced by {}. Metric space dilated.", dt); return Ok((AlgebraElement::scalar(1.0), BTreeSet::new())); }
                        "dump_geometry" => {
                            if args.len() != 1 { return Err("dump_geometry expects 1 string path".to_string()); }
                            let path = match &args[0] { Expr::Str(s) => s.clone(), Expr::Var(n) => strings.get(n).cloned().ok_or(format!("String variable {} not found", n))?, _ => return Err("dump_geometry requires a string path".to_string()) };
                            let events: Vec<EventId> = vm.graph.events.keys().cloned().collect();
                            let mut nodes_json = String::from("[");
                            for (i, eid) in events.iter().enumerate() { if i > 0 { nodes_json.push(','); } if let Some(ev) = vm.graph.events.get(eid) { let label = ev.label.replace('"', "'").replace("\\", "/"); let burden = ev.burden; let scale = ev.attentional_scale; let depth = ev.depth; nodes_json.push_str(&format!("{{\"id\":{},\"label\":\"{}\",\"burden\":{},\"scale\":{},\"depth\":{}}}", eid.0, label, burden, scale, depth)); } }
                            nodes_json.push(']');
                            let mut links_json = String::from("[");
                            let mut first_link = true;
                            for i in 0..events.len() { for j in (i+1)..events.len() { let elem_a = &vm.graph.events[&events[i]].element; let elem_b = &vm.graph.events[&events[j]].element; let k = vm.state.correlation_kernel(elem_a, elem_b); let scale_a = vm.graph.events[&events[i]].attentional_scale; let scale_b = vm.graph.events[&events[j]].attentional_scale; let metric_expansion = 1.0 / (scale_a * scale_b + 1e-12); let d_omega = -k.ln(); let d_eff = d_omega * metric_expansion; if d_eff < 10.0 { if !first_link { links_json.push(','); } first_link = false; links_json.push_str(&format!("{{\"source\":{},\"target\":{},\"distance\":{}}}", events[i].0, events[j].0, d_eff)); } } }
                            links_json.push(']');
                            let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RCL Emergent Geometry (Metric Expansion)</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #000; color: #fff; font-family: monospace; }}
        #info {{ position: absolute; top: 10px; left: 10px; z-index: 100; display: none; background: rgba(0,0,0,0.7); padding: 5px; border: 1px solid #0f0; }}
        canvas {{ display: block; }}
    </style>
</head>
<body>
    <div id="info">Node: <span id="node-label"></span><br>Burden: <span id="node-burden"></span><br>Scale: <span id="node-scale"></span></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const data = {{ "nodes": {}, "links": {} }};
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        camera.position.set(0, 0, 40);
        const nodes = []; const lines = []; const nodeMap = {{}};
        data.nodes.forEach(nodeData => {{
            const size = Math.max(0.2, 1.5 * (nodeData.scale || 1.0));
            const geometry = new THREE.SphereGeometry(size, 16, 16);
            let color = new THREE.Color(0x00ffff);
            if (nodeData.label.includes("archetype") || nodeData.label.includes("Sentiment")) {{ color.setHex(0xffff00); }}
            else {{ const scale = nodeData.scale || 1.0; color.setRGB(1.0 - scale, 0.5 * scale, scale); }}
            const material = new THREE.MeshBasicMaterial({{ color: color, transparent: true, opacity: Math.max(0.3, nodeData.scale || 1.0) }});
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set((Math.random() - 0.5) * 40, (Math.random() - 0.5) * 40, (Math.random() - 0.5) * 40);
            sphere.userData = {{ id: nodeData.id, label: nodeData.label, burden: nodeData.burden, scale: nodeData.scale, links: [] }};
            scene.add(sphere); nodes.push(sphere); nodeMap[nodeData.id] = sphere;
        }});
        const lineMaterial = new THREE.LineBasicMaterial({{ color: 0x00ff00, transparent: true, opacity: 0.3 }});
        data.links.forEach(link => {{
            if (!nodeMap[link.source] || !nodeMap[link.target]) return;
            const geometry = new THREE.BufferGeometry();
            const points = [ nodeMap[link.source].position, nodeMap[link.target].position ];
            geometry.setFromPoints(points);
            const line = new THREE.Line(geometry, lineMaterial);
            line.userData = {{ source: nodeMap[link.source], target: nodeMap[link.target] }};
            scene.add(line); lines.push(line);
            nodeMap[link.source].userData.links.push({{ target: nodeMap[link.target], dist: link.distance }});
            nodeMap[link.target].userData.links.push({{ target: nodeMap[link.source], dist: link.distance }});
        }});
        function simulate() {{
            nodes.forEach(node => {{
                if (!node.userData.links) return;
                node.userData.links.forEach(link => {{
                    const direction = new THREE.Vector3().subVectors(link.target.position, node.position);
                    let distance = direction.length();
                    if (distance === 0) distance = 0.1;
                    direction.normalize();
                    const stableLinkDist = Math.max(0.01, link.dist);
                    const targetDist = stableLinkDist * 5.0;
                    const force = (distance - targetDist) * 0.04;
                    node.position.add(direction.multiplyScalar(force));
                }});
            }});
            for (let i = 0; i < nodes.length; i++) {{
                for (let j = i + 1; j < nodes.length; j++) {{
                    const dir = new THREE.Vector3().subVectors(nodes[j].position, nodes[i].position);
                    let dist = dir.length();
                    if (dist === 0) dist = 0.1;
                    if (dist < 10.0) {{
                        dir.normalize();
                        const repelForce = (10.0 - dist) * 0.01;
                        nodes[i].position.sub(dir.clone().multiplyScalar(repelForce));
                        nodes[j].position.add(dir.clone().multiplyScalar(repelForce));
                    }}
                }}
            }}
        }}
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const infoBox = document.getElementById('info');
        const labelSpan = document.getElementById('node-label');
        const burdenSpan = document.getElementById('node-burden');
        const scaleSpan = document.getElementById('node-scale');
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(nodes);
            if (intersects.length > 0) {{
                infoBox.style.display = 'block';
                labelSpan.textContent = intersects[0].object.userData.label;
                burdenSpan.textContent = intersects[0].object.userData.burden.toFixed(4);
                scaleSpan.textContent = intersects[0].object.userData.scale.toFixed(4);
            }} else {{ infoBox.style.display = 'none'; }}
        }}
        window.addEventListener('mousemove', onMouseMove, false);
        function animate() {{
            requestAnimationFrame(animate);
            for(let i=0; i<5; i++) simulate();
            lines.forEach(line => {{
                const posAttr = line.geometry.attributes.position;
                posAttr.setXYZ(0, line.userData.source.position.x, line.userData.source.position.y, line.userData.source.position.z);
                posAttr.setXYZ(1, line.userData.target.position.x, line.userData.target.position.y, line.userData.target.position.z);
                posAttr.needsUpdate = true;
            }});
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"#, nodes_json, links_json);
                            std::fs::write(&path, html).map_err(|e| format!("Failed to write geometry: {}", e))?;
                            println!("  [Substrate Geometry] Emitted self-contained 3D topology to {}", path);
                            return Ok((AlgebraElement::scalar(1.0), BTreeSet::new()));
                        }
                        _ => {}
                    }
                    if let Some(ctmpl) = self.constraints.get(name).cloned() {
                        let mut arg_vals = Vec::new(); let mut prereqs = BTreeSet::new();
                        for arg in args { let (val, p) = self.eval_expr(arg, vm, scope, strings)?; arg_vals.push(val); prereqs.extend(p); }
                        if arg_vals.len() != ctmpl.params.len() { return Err(format!("Constraint {} expects {} args", name, ctmpl.params.len())); }
                        let mut constraint_scope = scope.clone();
                        for ((pname, _), aval) in ctmpl.params.iter().zip(arg_vals.iter()) { constraint_scope.insert(pname.clone(), aval.clone()); }
                        let (cond_val, _) = self.eval_expr(&ctmpl.condition, vm, &constraint_scope, strings)?;
                        return Ok((cond_val, prereqs));
                    }
                    let template = self.events.get(name).ok_or(format!("Unknown event: {}", name))?.clone();
                    let mut local_scope = scope.clone();
                    let mut local_strings = strings.clone();

                    // 1. Correctly populate string parameters from BOTH literals and string variables
                    for (i, arg) in args.iter().enumerate() {
                        let param_name = &template.params[i].0;
                        match arg {
                            Expr::Str(s) => {
                                local_strings.insert(param_name.clone(), s.clone());
                            }
                            Expr::Var(n) => {
                                if let Some(s) = strings.get(n).or_else(|| self.let_strings.get(n)) {
                                    local_strings.insert(param_name.clone(), s.clone());
                                }
                            }
                            _ => {}
                        }
                    }

                    let mut arg_vals = Vec::new(); 
                    let mut prereqs = BTreeSet::new();

                    // 2. Prevent algebraic evaluation if the argument resolves to a string context
                    for (i, arg) in args.iter().enumerate() {
                        let is_string_ctx = match arg {
                            Expr::Str(_) => true,
                            Expr::Var(n) => strings.contains_key(n) || self.let_strings.contains_key(n),
                            _ => false,
                        };

                        if !is_string_ctx { 
                            let (val, p) = self.eval_expr(arg, vm, &local_scope, &local_strings)?; 
                            arg_vals.push(val); 
                            prereqs.extend(p); 
                        } else { 
                            // Pad the numerical vector with zero for string slots
                            arg_vals.push(AlgebraElement::zero()); 
                        }
                    }

                    // 3. Bind algebraic values to local scope, skipping string contexts
                    for (i, ((pname, _), aval)) in template.params.iter().zip(arg_vals.iter()).enumerate() {
                        let is_string_ctx = match &args[i] {
                            Expr::Str(_) => true,
                            Expr::Var(n) => strings.contains_key(n) || self.let_strings.contains_key(n),
                            _ => false,
                        };
                        if !is_string_ctx {
                            local_scope.insert(pname.clone(), aval.clone());
                        }
                    }

                    for req in &template.requires { let (cond_val, _) = self.eval_expr(req, vm, &local_scope, &local_strings)?; if let Some(scalar) = cond_val.as_scalar() { if scalar <= 0.0 { return Err(format!("Requires clause violated: {}", name)); } } else { return Err("Requires clause must be scalar".to_string()); } }
                    let (result, body_prereqs) = self.eval_expr(&template.body, vm, &local_scope, &local_strings)?; prereqs.extend(body_prereqs);
                    let eid = vm.execute(result.clone(), prereqs, format!("{}(...)", name))?;
                    let mut new_prereqs = BTreeSet::new(); new_prereqs.insert(eid);
                    Ok((result, new_prereqs))
                }
                Expr::Match(scrutinee, arms) => { let (sv, mut prereqs) = self.eval_expr(scrutinee, vm, scope, strings)?; let ss = sv.as_scalar().ok_or("Match scrutinee must be scalar")?; for arm in arms { if arm.is_wildcard { let (val, p) = self.eval_expr(&arm.body, vm, scope, strings)?; prereqs.extend(p); return Ok((val, prereqs)); } let (pv, _) = self.eval_expr(&arm.pattern, vm, scope, strings)?; if let Some(ps) = pv.as_scalar() { if (ss - ps).abs() < 1e-10 { let (val, p) = self.eval_expr(&arm.body, vm, scope, strings)?; prereqs.extend(p); return Ok((val, prereqs)); } } } Err("No matching case".to_string()) }
                Expr::Str(_) => Err("String literals not supported in pure relational math".to_string()),
                Expr::Block(stmts, expr) => {
                    let mut local_scope = scope.clone(); let mut local_strings = strings.clone(); let mut prereqs = BTreeSet::new();
                    for stmt in stmts {
                        match stmt {
                            Stmt::Let { name, value } => { if let Expr::Str(s) = value { local_strings.insert(name.clone(), s.clone()); } else { let (elem, p) = self.eval_expr(value, vm, &local_scope, &local_strings)?; prereqs.extend(p); local_scope.insert(name.clone(), elem); } }
                            Stmt::Print(e) => { let (elem, _) = self.eval_expr(e, vm, &local_scope, &local_strings)?; if let Some(scalar) = elem.as_scalar() { println!("Print: {}", scalar); } else { let val = vm.state.gns_vector(&elem); println!("Print profile [Vector Size: {} -> Norm: {:.4}]", val.len(), norm_sq(&val)); } }
                            Stmt::Expr(e) => { let (_, p) = self.eval_expr(e, vm, &local_scope, &local_strings)?; prereqs.extend(p); }
                            Stmt::Ingest(text) => { let _ = self.ingest_text(text, vm)?; }
                        }
                    }
                    let (val, p) = self.eval_expr(expr, vm, &local_scope, &local_strings)?; prereqs.extend(p); Ok((val, prereqs))
                }
            }
        }
    }
    fn is_keyword(word: &str) -> bool { matches!(word, "gen" | "let" | "concept" | "constraint" | "event" | "if" | "else" | "match" | "ingest" | "print" | "enforce" | "requires" | "fn" | "import") }
}

mod memory {
    use super::vm::*; use super::compiler::Compiler; use super::algebra::*; use std::io::{Read, Write}; use std::collections::BTreeSet;
    pub struct MemoryStore { pub vm: RCSVM, pub compiler: Compiler }
    impl MemoryStore {
        pub fn save(&self, path: &str) -> std::io::Result<()> { let mut file = std::fs::File::create(path)?; file.write_all(&self.vm.state.basis_dim.to_le_bytes())?; file.write_all(&self.vm.next_gen.to_le_bytes())?; file.write_all(&self.vm.next_constraint.to_le_bytes())?; file.write_all(&(self.vm.state.omega_vector.len() as u64).to_le_bytes())?; for c in &self.vm.state.omega_vector { file.write_all(&c.re.to_le_bytes())?; file.write_all(&c.im.to_le_bytes())?; } file.write_all(&(self.vm.graph.events.len() as u64).to_le_bytes())?; for (id, ev) in &self.vm.graph.events { file.write_all(&id.0.to_le_bytes())?; file.write_all(&ev.burden.to_le_bytes())?; file.write_all(&ev.depth.to_le_bytes())?; file.write_all(&(ev.label.len() as u64).to_le_bytes())?; file.write_all(ev.label.as_bytes())?; } Ok(()) }
        pub fn load(path: &str) -> std::io::Result<Self> { let mut file = std::fs::File::open(path)?; let mut buf = [0u8; 8]; file.read_exact(&mut buf)?; let basis_dim = u64::from_le_bytes(buf) as usize; file.read_exact(&mut buf)?; let next_gen = u64::from_le_bytes(buf); file.read_exact(&mut buf)?; let next_constraint = u64::from_le_bytes(buf); let mut store = MemoryStore { vm: RCSVM::new(basis_dim), compiler: Compiler::new() }; store.vm.next_gen = next_gen; store.vm.next_constraint = next_constraint; file.read_exact(&mut buf)?; let omega_len = u64::from_le_bytes(buf) as usize; for _ in 0..omega_len { file.read_exact(&mut buf)?; let _re = f64::from_le_bytes(buf); file.read_exact(&mut buf)?; let _im = f64::from_le_bytes(buf); } file.read_exact(&mut buf)?; let event_count = u64::from_le_bytes(buf) as usize; for _ in 0..event_count { file.read_exact(&mut buf)?; let id = u64::from_le_bytes(buf); file.read_exact(&mut buf)?; let burden = f64::from_le_bytes(buf); file.read_exact(&mut buf)?; let depth = u64::from_le_bytes(buf) as usize; file.read_exact(&mut buf)?; let lbl_len = u64::from_le_bytes(buf) as usize; let mut lbl_buf = vec![0u8; lbl_len]; file.read_exact(&mut lbl_buf)?; let label = String::from_utf8(lbl_buf).unwrap_or_default(); let eid = EventId(id); store.vm.graph.events.insert(eid, CausalEvent { id: eid, element: AlgebraElement::zero(), closure_set: BTreeSet::new(), prereqs: BTreeSet::new(), burden, attentional_scale: 1.0, depth, label }); store.vm.graph.next_id = store.vm.graph.next_id.max(id + 1); } Ok(store) }
        pub fn load_from_bytes(data: &[u8]) -> std::io::Result<Self> {
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 8];
            
            cursor.read_exact(&mut buf)?; let basis_dim = u64::from_le_bytes(buf) as usize;
            cursor.read_exact(&mut buf)?; let next_gen = u64::from_le_bytes(buf);
            cursor.read_exact(&mut buf)?; let next_constraint = u64::from_le_bytes(buf);
            
            let mut store = MemoryStore { vm: RCSVM::new(basis_dim), compiler: Compiler::new() };
            store.vm.next_gen = next_gen;
            store.vm.next_constraint = next_constraint;
            
            cursor.read_exact(&mut buf)?; let omega_len = u64::from_le_bytes(buf) as usize;
            for _ in 0..omega_len { cursor.read_exact(&mut buf)?; let _re = f64::from_le_bytes(buf); cursor.read_exact(&mut buf)?; let _im = f64::from_le_bytes(buf); }
            
            cursor.read_exact(&mut buf)?; let event_count = u64::from_le_bytes(buf) as usize;
            for _ in 0..event_count {
                cursor.read_exact(&mut buf)?; let id = u64::from_le_bytes(buf);
                cursor.read_exact(&mut buf)?; let burden = f64::from_le_bytes(buf);
                cursor.read_exact(&mut buf)?; let depth = u64::from_le_bytes(buf) as usize;
                
                cursor.read_exact(&mut buf)?; let lbl_len = u64::from_le_bytes(buf) as usize;
                let mut lbl_buf = vec![0u8; lbl_len];
                cursor.read_exact(&mut lbl_buf)?;
                let label = String::from_utf8(lbl_buf).unwrap_or_default();
                
                let eid = EventId(id);
                store.vm.graph.events.insert(eid, CausalEvent { id: eid, element: AlgebraElement::zero(), closure_set: BTreeSet::new(), prereqs: BTreeSet::new(), burden, attentional_scale: 1.0, depth, label });
                store.vm.graph.next_id = store.vm.graph.next_id.max(id + 1);
            }
            Ok(store)
        }
    }
}

fn run_program(source: &str) {
    let mut store = if std::path::Path::new("memory.bin").exists() {
        memory::MemoryStore::load("memory.bin").unwrap_or_else(|_| {
            let mut s = memory::MemoryStore { vm: vm::RCSVM::new(1), compiler: compiler::Compiler::new() };
            let _ = s.compiler.compile(&crate::parser::Parser::new(crate::lexer::Lexer::new(CORE_SOURCE).tokenize().unwrap()).parse_program().unwrap(), &mut s.vm);
            s
        })
    } else {
        let mut s = memory::MemoryStore { vm: vm::RCSVM::new(1), compiler: compiler::Compiler::new() };
        let _ = s.compiler.compile(&crate::parser::Parser::new(crate::lexer::Lexer::new(CORE_SOURCE).tokenize().unwrap()).parse_program().unwrap(), &mut s.vm);
        s
    };

    let tokens = match lexer::Lexer::new(source).tokenize() {
        Ok(t) => t,
        Err(e) => { eprintln!("Lex error: {}", e); return; }
    };

    let program = match parser::Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };

    match store.compiler.compile(&program, &mut store.vm) {
        Ok(()) => {
            println!("\n═══ Execution Complete ═══");
            let _ = store.save("memory.bin");
        }
        Err(e) => eprintln!("Runtime error: {}", e),
    }
}

fn repl() {
    let path = "memory.bin";
    let mut store = if std::path::Path::new("memory.bin").exists() {
        memory::MemoryStore::load("memory.bin").unwrap_or_else(|_| {
            let mut s = memory::MemoryStore { vm: vm::RCSVM::new(1), compiler: compiler::Compiler::new() };
            let _ = s.compiler.compile(&crate::parser::Parser::new(crate::lexer::Lexer::new(CORE_SOURCE).tokenize().unwrap()).parse_program().unwrap(), &mut s.vm);
            s
        })
    } else {
        let mut s = memory::MemoryStore { vm: vm::RCSVM::new(1), compiler: compiler::Compiler::new() };
        let _ = s.compiler.compile(&crate::parser::Parser::new(crate::lexer::Lexer::new(CORE_SOURCE).tokenize().unwrap()).parse_program().unwrap(), &mut s.vm);
        s
    };
    println!("╔══════════════════════════════════════════════╗");
    println!("║  RCL v1.0 Boot Kernel (Phase 8)              ║");
    println!("║  Module System & Composable Stdlib           ║");
    println!("╚══════════════════════════════════════════════╝");
    println!(); println!("Enter RCL code. Use ;; to submit a block."); println!("Type 'quit' to exit.");
    let stdin = io::stdin(); let mut buffer = String::new();
    loop { print!("rcl> "); io::stdout().flush().unwrap(); let mut line = String::new(); match stdin.lock().read_line(&mut line) { Ok(0) => break, Ok(_) => {}, Err(_) => break, } let trimmed = line.trim(); if trimmed == "quit" { let _ = store.save(path); break; } buffer.push_str(&line); if trimmed.ends_with(";;") { buffer = buffer.trim_end_matches(";;").to_string(); let tokens = match lexer::Lexer::new(&buffer).tokenize() { Ok(t) => t, Err(e) => { eprintln!("Lex error: {}", e); buffer.clear(); continue; } }; let program = match parser::Parser::new(tokens).parse_program() { Ok(p) => p, Err(e) => { eprintln!("Parse error: {}", e); buffer.clear(); continue; } }; match store.compiler.compile(&program, &mut store.vm) { Ok(()) => { if store.vm.is_admissible { println!("  ✓ admissible"); } else { println!("  ✗ admissibility violation"); } let _ = store.save(path); } Err(e) => eprintln!("  ✗ {}", e), } buffer.clear(); } }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        let filename = &args[1];
        match std::fs::read_to_string(filename) {
            Ok(source) => run_program(&source),
            Err(e) => eprintln!("Error reading {}: {}", filename, e),
        }
    } else {
        repl();
    }
}
