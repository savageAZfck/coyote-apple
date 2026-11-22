# REV_DEMO — video script: reversible computing as a language primitive

~3:45 runtime. Record the terminal (QuickTime / Screenshot toolbar, large
font, dark mode, notifications off). Every command below is real and
verified — the numbers on screen are the actual outputs.

Files: `examples/revdemo.cy` (the demo program), `examples/arx.cy`,
`examples/seal.cy`, `examples/arx_mv.cy`.

Binary under demo: `target/debug/coyote-apple` (build first:
`cargo build` in coyote-apple).

---

## Beat 1 — The claim (0:00–0:30)

**VO:** "Every program you've ever written destroys information. This
one doesn't. In Coyote, `rev fn` declares a bijective program — the
compiler derives its inverse, and the machine proves it. Here's an
8-round ARX cipher, running backwards, on the silicon already in your
laptop."

**Screen:** open `examples/revdemo.cy`. Pause on `rev host fn arx8`.
Cursor to `var x = st.x` / `var y = st.y` — "the state is explicit."
Scroll the 8 rounds — add, rotate, xor, key schedule.

## Beat 2 — Encrypt (0:30–0:55)

```
coyote-apple run examples/revdemo.cy
```

**Screen output:**

```
PLAINTEXT = 1234567890123456789 -9876543210987
CIPHERTEXT= 360642182225145438 -3841046212154815954
RECOVERED = 1234567890123456789 -9876543210987
roundtrip = 1
```

**VO:** "Encrypted. Now watch the part that shouldn't be possible —
the decryption was never written."

## Beat 3 — The inverse was never written (0:55–1:35)

**Screen:** scroll `main` — show `let pt = uncall arx8(ct, key);`.
There is no `arx8_inv` anywhere in the file.

**VO:** "`uncall` resolves to a compiler-derived inverse — statements
reversed in Bennett order, `+=` swaps with `-=`, `^=` is its own
inverse, `fi` predicates swap roles. That's Janus semantics, derived
mechanically. I wrote the cipher. The toolchain wrote its inverse.
And it recovered the plaintext bit-for-bit."

## Beat 4 — The gate doesn't trust the compiler (1:35–2:15)

```
coyote-apple revcheck examples/revdemo.cy
```

**Screen output:**

```
revcheck — examples/revdemo.cy (2 pairs, tol 0)
  ✓ step ↔ step_inv                      max|Δ| 0.000000
  ✓ arx8 ↔ arx8_inv                      max|Δ| 0.000000
✓ pass — inv∘f = id on every rev pair
```

**VO:** "`revck` is a structural check — it can't see floating point.
`revcheck` doesn't trust it: it runs the forward program, runs the
inverse on the output, and measures. It already caught a real bug —
mixing f32 `+=` and `^=` on tensors isn't bit-exact. Max diff 8.0.
The syntax looked reversible. The execution gate said no."

**Optional cutaway (strong beat, ~20s):** run
`coyote-apple revcheck /tmp/revtests/v1.cy` — shows the FAIL live:
`step ↔ step_inv  max|Δ| 8.000000 FAIL`. "That's the gate working —
not a green compile, a measurement."

## Beat 5 — The certificate (2:15–2:50)

```
coyote-apple certify examples/revdemo.cy -o revdemo.cert
coyote-apple certify --check revdemo.cert
```

**Screen:** cert output — `compiles: held`, `parity: held`,
`replay_determinism: held`, `reversible: held` → `CERTIFIED`.
On `--check`: all claims re-derived `held`.

**VO:** "The certificate binds five claims — compiles, parity across
substrates, replay determinism, and reversible. `--check` re-derives
every one from source. Change a single character and the certificate
is void. 'Prove the computation happened' isn't a forensic afterthought
— it's a primitive."

## Beat 6 — Both strata (2:50–3:15)

**Screen:** back in `revdemo.cy` — `rev host fn arx8` (host/i64) and
`rev fn step` (tensor/f16). During `run`, note the emit lines:
`✓ step_inv — rev backend`.

**VO:** "Same primitive, two strata. `rev host fn` for scalars and
control flow — that's where real ARX lives. `rev fn` for tensor
programs — compiled to the reversible backend. One language,
reversible wherever it can be."

## Beat 7 — What it means (3:15–3:45)

**VO:** "This is reversible computing as a language primitive — not
a library, not a convention, not a research prototype. Safety is
compile-time. Inversion is compiler-derived. Correctness is
machine-verified and certifiable. Janus proved the theory in 1982 —
this runs on the silicon already in your pocket. The language core
is MIT and free. Coyote-Apple is the layer that makes it sovereign.
Your silicon, your weights, your proofs — and now, your programs
can run backwards."

**End card:**
- github.com/savageAZfck/coyote (MIT core)
- github.com/savageAZfck/coyote-apple (BSL product)
- `coyote-apple revcheck <file.cy>` — the inversion gate
- `coyote-apple certify <file.cy>` — the certificate

---

## Recording checklist

- [ ] `cargo build` in coyote, then coyote-apple (debug binaries are fine)
- [ ] Terminal: large monospace font, dark background, `cd` into repo root
- [ ] Run each command once beforehand so emit caches are warm (faster take)
- [ ] For Beat 4 cutaway: `cp /tmp/revtests/v1.cy .` or use a path that exists
- [ ] Pause ~1s on each output block before continuing
