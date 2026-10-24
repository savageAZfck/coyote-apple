// std/expect — test assertions for `host fn test_*` cases.
// `import "std/expect"` (CYO_PATH ships coyote-apple's lib/).
//
// Convention: return 0 on pass, 1 on fail — a test fn returns the
// expect result, and `coyote-apple test` counts the exit.

host fn expect_eq(a: i64, b: i64) -> i64 {
    if a == b {
        return 0;
    }
    print("  expected", b, "got", a);
    return 1;
}

host fn expect_true(c: bool) -> i64 {
    if c {
        return 0;
    }
    print("  expected true, got false");
    return 1;
}

host fn expect_ne(a: i64, b: i64) -> i64 {
    if a != b {
        return 0;
    }
    print("  expected !=", b, "got", a);
    return 1;
}

host fn expect_near(a: f64, b: f64, tol: f64) -> i64 {
    var d = a - b;
    if d < 0.0 {
        d = 0.0 - d;
    }
    if d <= tol {
        return 0;
    }
    print("  expected", b, "±", tol, "got", a);
    return 1;
}
