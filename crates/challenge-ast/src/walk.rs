//! AST feature extraction (no visitor feature — explicit match walk).

use std::collections::{BTreeMap, BTreeSet};

use rustpython_parser::ast::{self, Expr, Int, Stmt};

pub(crate) struct Features {
    pub node_counts: BTreeMap<String, u32>,
    pub imports: BTreeSet<String>,
    pub defs: BTreeSet<String>,
    pub calls: BTreeSet<String>,
    pub structure_tokens: Vec<String>,
}

pub(crate) fn collect_features(suite: &[Stmt]) -> Features {
    let mut f = Features {
        node_counts: BTreeMap::new(),
        imports: BTreeSet::new(),
        defs: BTreeSet::new(),
        calls: BTreeSet::new(),
        structure_tokens: Vec::new(),
    };
    for stmt in suite {
        walk_stmt(&mut f, stmt);
    }
    f
}

fn bump(f: &mut Features, name: &str) {
    f.structure_tokens.push(name.to_owned());
    *f.node_counts.entry(name.to_owned()).or_insert(0) += 1;
}

#[allow(clippy::too_many_lines)] // exhaustive Stmt match; splitting hurts clarity
fn walk_stmt(f: &mut Features, stmt: &Stmt) {
    match stmt {
        Stmt::FunctionDef(n) => {
            bump(f, "FunctionDef");
            f.defs.insert(n.name.to_string());
            for d in &n.decorator_list {
                walk_expr(f, d);
            }
            walk_args(f, &n.args);
            if let Some(r) = &n.returns {
                walk_expr(f, r);
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
        Stmt::AsyncFunctionDef(n) => {
            bump(f, "AsyncFunctionDef");
            f.defs.insert(n.name.to_string());
            for d in &n.decorator_list {
                walk_expr(f, d);
            }
            walk_args(f, &n.args);
            if let Some(r) = &n.returns {
                walk_expr(f, r);
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
        Stmt::ClassDef(n) => {
            bump(f, "ClassDef");
            f.defs.insert(n.name.to_string());
            for d in &n.decorator_list {
                walk_expr(f, d);
            }
            for b in &n.bases {
                walk_expr(f, b);
            }
            for k in &n.keywords {
                walk_expr(f, &k.value);
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
        Stmt::Return(n) => {
            bump(f, "Return");
            if let Some(v) = &n.value {
                walk_expr(f, v);
            }
        }
        Stmt::Delete(n) => {
            bump(f, "Delete");
            for t in &n.targets {
                walk_expr(f, t);
            }
        }
        Stmt::Assign(n) => {
            bump(f, "Assign");
            for t in &n.targets {
                walk_expr(f, t);
            }
            walk_expr(f, &n.value);
        }
        Stmt::TypeAlias(n) => {
            bump(f, "TypeAlias");
            walk_expr(f, &n.name);
            walk_expr(f, &n.value);
        }
        Stmt::AugAssign(n) => {
            bump(f, "AugAssign");
            walk_expr(f, &n.target);
            walk_expr(f, &n.value);
        }
        Stmt::AnnAssign(n) => {
            bump(f, "AnnAssign");
            walk_expr(f, &n.target);
            walk_expr(f, &n.annotation);
            if let Some(v) = &n.value {
                walk_expr(f, v);
            }
        }
        Stmt::For(n) => {
            bump(f, "For");
            walk_expr(f, &n.target);
            walk_expr(f, &n.iter);
            for s in &n.body {
                walk_stmt(f, s);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
        }
        Stmt::AsyncFor(n) => {
            bump(f, "AsyncFor");
            walk_expr(f, &n.target);
            walk_expr(f, &n.iter);
            for s in &n.body {
                walk_stmt(f, s);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
        }
        Stmt::While(n) => {
            bump(f, "While");
            walk_expr(f, &n.test);
            for s in &n.body {
                walk_stmt(f, s);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
        }
        Stmt::If(n) => {
            bump(f, "If");
            walk_expr(f, &n.test);
            for s in &n.body {
                walk_stmt(f, s);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
        }
        Stmt::With(n) => {
            bump(f, "With");
            for item in &n.items {
                walk_expr(f, &item.context_expr);
                if let Some(v) = &item.optional_vars {
                    walk_expr(f, v);
                }
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
        Stmt::AsyncWith(n) => {
            bump(f, "AsyncWith");
            for item in &n.items {
                walk_expr(f, &item.context_expr);
                if let Some(v) = &item.optional_vars {
                    walk_expr(f, v);
                }
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
        Stmt::Match(n) => {
            bump(f, "Match");
            walk_expr(f, &n.subject);
            for case in &n.cases {
                if let Some(g) = &case.guard {
                    walk_expr(f, g);
                }
                for s in &case.body {
                    walk_stmt(f, s);
                }
            }
        }
        Stmt::Raise(n) => {
            bump(f, "Raise");
            if let Some(e) = &n.exc {
                walk_expr(f, e);
            }
            if let Some(c) = &n.cause {
                walk_expr(f, c);
            }
        }
        Stmt::Try(n) => {
            bump(f, "Try");
            for s in &n.body {
                walk_stmt(f, s);
            }
            for h in &n.handlers {
                walk_excepthandler(f, h);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
            for s in &n.finalbody {
                walk_stmt(f, s);
            }
        }
        Stmt::TryStar(n) => {
            bump(f, "TryStar");
            for s in &n.body {
                walk_stmt(f, s);
            }
            for h in &n.handlers {
                walk_excepthandler(f, h);
            }
            for s in &n.orelse {
                walk_stmt(f, s);
            }
            for s in &n.finalbody {
                walk_stmt(f, s);
            }
        }
        Stmt::Assert(n) => {
            bump(f, "Assert");
            walk_expr(f, &n.test);
            if let Some(m) = &n.msg {
                walk_expr(f, m);
            }
        }
        Stmt::Import(n) => {
            bump(f, "Import");
            for a in &n.names {
                f.imports.insert(a.name.to_string());
            }
        }
        Stmt::ImportFrom(n) => {
            bump(f, "ImportFrom");
            let mod_name = n.module.as_ref().map_or_else(
                || {
                    let level = n.level.as_ref().map_or(0, Int::to_usize);
                    ".".repeat(level.max(1))
                },
                ToString::to_string,
            );
            f.imports.insert(mod_name);
        }
        Stmt::Global(n) => {
            bump(f, "Global");
            for name in &n.names {
                f.defs.insert(name.to_string());
            }
        }
        Stmt::Nonlocal(n) => {
            bump(f, "Nonlocal");
            for name in &n.names {
                f.defs.insert(name.to_string());
            }
        }
        Stmt::Expr(n) => {
            bump(f, "Expr");
            walk_expr(f, &n.value);
        }
        Stmt::Pass(_) => bump(f, "Pass"),
        Stmt::Break(_) => bump(f, "Break"),
        Stmt::Continue(_) => bump(f, "Continue"),
    }
}

fn walk_excepthandler(f: &mut Features, h: &ast::ExceptHandler) {
    match h {
        ast::ExceptHandler::ExceptHandler(n) => {
            bump(f, "ExceptHandler");
            if let Some(t) = &n.type_ {
                walk_expr(f, t);
            }
            for s in &n.body {
                walk_stmt(f, s);
            }
        }
    }
}

fn walk_args(f: &mut Features, args: &ast::Arguments) {
    bump(f, "Arguments");
    for a in args
        .posonlyargs
        .iter()
        .chain(args.args.iter())
        .chain(args.kwonlyargs.iter())
    {
        if let Some(ann) = &a.def.annotation {
            walk_expr(f, ann);
        }
        if let Some(d) = &a.default {
            walk_expr(f, d);
        }
    }
    if let Some(v) = &args.vararg {
        if let Some(ann) = &v.annotation {
            walk_expr(f, ann);
        }
    }
    if let Some(v) = &args.kwarg {
        if let Some(ann) = &v.annotation {
            walk_expr(f, ann);
        }
    }
}

#[allow(clippy::too_many_lines)] // exhaustive Expr match; splitting hurts clarity
fn walk_expr(f: &mut Features, expr: &Expr) {
    match expr {
        Expr::BoolOp(n) => {
            bump(f, "BoolOp");
            for v in &n.values {
                walk_expr(f, v);
            }
        }
        Expr::NamedExpr(n) => {
            bump(f, "NamedExpr");
            walk_expr(f, &n.target);
            walk_expr(f, &n.value);
        }
        Expr::BinOp(n) => {
            bump(f, "BinOp");
            walk_expr(f, &n.left);
            walk_expr(f, &n.right);
        }
        Expr::UnaryOp(n) => {
            bump(f, "UnaryOp");
            walk_expr(f, &n.operand);
        }
        Expr::Lambda(n) => {
            bump(f, "Lambda");
            walk_args(f, &n.args);
            walk_expr(f, &n.body);
        }
        Expr::IfExp(n) => {
            bump(f, "IfExp");
            walk_expr(f, &n.test);
            walk_expr(f, &n.body);
            walk_expr(f, &n.orelse);
        }
        Expr::Dict(n) => {
            bump(f, "Dict");
            for k in n.keys.iter().flatten() {
                walk_expr(f, k);
            }
            for v in &n.values {
                walk_expr(f, v);
            }
        }
        Expr::Set(n) => {
            bump(f, "Set");
            for e in &n.elts {
                walk_expr(f, e);
            }
        }
        Expr::ListComp(n) => {
            bump(f, "ListComp");
            walk_expr(f, &n.elt);
            for g in &n.generators {
                walk_comprehension(f, g);
            }
        }
        Expr::SetComp(n) => {
            bump(f, "SetComp");
            walk_expr(f, &n.elt);
            for g in &n.generators {
                walk_comprehension(f, g);
            }
        }
        Expr::DictComp(n) => {
            bump(f, "DictComp");
            walk_expr(f, &n.key);
            walk_expr(f, &n.value);
            for g in &n.generators {
                walk_comprehension(f, g);
            }
        }
        Expr::GeneratorExp(n) => {
            bump(f, "GeneratorExp");
            walk_expr(f, &n.elt);
            for g in &n.generators {
                walk_comprehension(f, g);
            }
        }
        Expr::Await(n) => {
            bump(f, "Await");
            walk_expr(f, &n.value);
        }
        Expr::Yield(n) => {
            bump(f, "Yield");
            if let Some(v) = &n.value {
                walk_expr(f, v);
            }
        }
        Expr::YieldFrom(n) => {
            bump(f, "YieldFrom");
            walk_expr(f, &n.value);
        }
        Expr::Compare(n) => {
            bump(f, "Compare");
            walk_expr(f, &n.left);
            for c in &n.comparators {
                walk_expr(f, c);
            }
        }
        Expr::Call(n) => {
            bump(f, "Call");
            if let Some(name) = call_name(&n.func) {
                f.calls.insert(name);
            }
            walk_expr(f, &n.func);
            for a in &n.args {
                walk_expr(f, a);
            }
            for k in &n.keywords {
                walk_expr(f, &k.value);
            }
        }
        Expr::FormattedValue(n) => {
            bump(f, "FormattedValue");
            walk_expr(f, &n.value);
            if let Some(fmt) = &n.format_spec {
                walk_expr(f, fmt);
            }
        }
        Expr::JoinedStr(n) => {
            bump(f, "JoinedStr");
            for v in &n.values {
                walk_expr(f, v);
            }
        }
        Expr::Constant(_) => bump(f, "Constant"),
        Expr::Attribute(n) => {
            bump(f, "Attribute");
            walk_expr(f, &n.value);
        }
        Expr::Subscript(n) => {
            bump(f, "Subscript");
            walk_expr(f, &n.value);
            walk_expr(f, &n.slice);
        }
        Expr::Starred(n) => {
            bump(f, "Starred");
            walk_expr(f, &n.value);
        }
        Expr::Name(_) => bump(f, "Name"),
        Expr::List(n) => {
            bump(f, "List");
            for e in &n.elts {
                walk_expr(f, e);
            }
        }
        Expr::Tuple(n) => {
            bump(f, "Tuple");
            for e in &n.elts {
                walk_expr(f, e);
            }
        }
        Expr::Slice(n) => {
            bump(f, "Slice");
            if let Some(l) = &n.lower {
                walk_expr(f, l);
            }
            if let Some(u) = &n.upper {
                walk_expr(f, u);
            }
            if let Some(s) = &n.step {
                walk_expr(f, s);
            }
        }
    }
}

fn walk_comprehension(f: &mut Features, g: &ast::Comprehension) {
    bump(f, "Comprehension");
    walk_expr(f, &g.target);
    walk_expr(f, &g.iter);
    for i in &g.ifs {
        walk_expr(f, i);
    }
}

fn call_name(func: &Expr) -> Option<String> {
    match func {
        Expr::Name(n) => Some(n.id.to_string()),
        Expr::Attribute(n) => Some(n.attr.to_string()),
        _ => None,
    }
}
