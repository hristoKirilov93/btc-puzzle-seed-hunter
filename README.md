# 🧩 BTC Puzzle Seed Hunter — GPU deterministic-wallet searcher (`gpuseed`) - If found, you get access to all unsolved puzzles ~903 BTC

> Hunting the Bitcoin Puzzle creator's **Electrum v1 wallet seed** on the GPU —
> every candidate seed generated, stretched, turned into a wallet and checked
> against the solved puzzles, entirely on the card.
>
> Built by **Hristo × Claude**.

A fully GPU-resident searcher over the **2¹²⁸ Electrum v1 old-seed space**. Every
seed is generated on the GPU, stretched through Electrum's 100 000-round
SHA-256 KDF, turned into a master key (`a·G`), and run through a cascade that
checks it against the known Bitcoin Puzzle residues — all on the card. The CPU
only drains the (rare) hits, so it can run next to a separate CPU hunter
(e.g. ecloop) without competing for cores.

It targets **[Bitcoin Puzzle #71](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)**
(address `1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU`), but the cascade verifies against
all solved puzzles #1–#135, so a genuine wallet hit is detected with a false-hit
rate around 2⁻⁵⁰⁰.

> **Scope, stated honestly.** This tool is a bet that the puzzle creator used an
> old Electrum (v1) wallet. It searches the *seed* space (2¹²⁸), which is far
> larger than #71's raw key space (2⁷⁰). It exists because the *structured* seed
> hypotheses were already exhausted (see **What we tested**); this is the
> brute-random fallback. Read that section before you spend GPU hours.

---

## 1. Quick start

### Requirements
- NVIDIA GPU + CUDA toolkit (`nvcc`). Tested on an RTX 4060 Ti (Ada, `sm_89`).
- OpenSSL dev headers (`libssl-dev`) and OpenMP (`libgomp`).
- The files in this repo must sit together: `gpuseed.cu`, `known_keys.h`,
  `hitcrypt.h`, `old_words.txt`.

On Ubuntu / WSL:
```bash
sudo apt install nvidia-cuda-toolkit libssl-dev
```

### Build
```bash
# RTX 40xx / Ada:
nvcc -O3 -arch=sm_89  -Xcompiler -fopenmp gpuseed.cu -o gpuseed -lcrypto -lgomp
# RTX 50xx / Blackwell:
nvcc -O3 -arch=sm_120 -Xcompiler -fopenmp gpuseed.cu -o gpuseed -lcrypto -lgomp
```
Or just `make` (override the arch with `make ARCH=sm_120`).

### Verify before you trust any result
```bash
./gpuseed --selftest
```
Expected:
```
GPU stretch vector:  OK
GPU a*G (MPK):       OK
GPU child(0,0):      OK
P2PKH address test:  OK
mnemonic (12 words): OK
SELFTEST PASS
```
If this does not say `SELFTEST PASS`, do not rely on the search — open an issue
with your GPU, driver and CUDA version.

### Run
```bash
CUDA_VISIBLE_DEVICES=0 ./gpuseed
```
It prints the starting base and a live rate:
```
gpuseed GPU-only | dev=NVIDIA GeForce RTX 4060 Ti sm_89 | targets=83 probes=6 batch=1048576
new base=...  (progress -> gpuseed_progress.txt)
tested 1048576 | 24966 seeds/s | hits 0
```

### Options (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `BATCH` | `1048576` | Seeds per GPU batch. Bigger = fewer/longer bursts, more VRAM (`BATCH×32` bytes). Does **not** change throughput — the card saturates at ~50k threads. |
| `JUMP` | off | `JUMP=1` picks a fresh random base every batch (uniform sampling of the whole space) instead of a contiguous sweep. |
| `PROGRESS` | `gpuseed_progress.txt` | Progress file for auto-resume. Give each GPU its own file for multi-card runs. |
| `BASE_HI`,`BASE_LO` | — | Start from an explicit 128-bit base (two 64-bit hex halves). Overrides the progress file. |
| `HIT_PUBKEY` | — | PEM public key; on a hit the secret is RSA-OAEP encrypted to `hit.enc` instead of written in clear. Use this on rented machines. |
| `HIT_OUT` | `hit.enc` | Output file for the encrypted hit. |

### Resume
With the default (contiguous) mode, the next base is written to the progress
file after every batch. Stop with Ctrl-C and re-run — it continues from where it
left off. Delete the progress file to start fresh. `BASE_HI/BASE_LO` ignore it.

### Multi-GPU
One process per card, each with its own progress file:
```bash
PROGRESS=prog0.txt CUDA_VISIBLE_DEVICES=0 ./gpuseed &
PROGRESS=prog1.txt CUDA_VISIBLE_DEVICES=1 ./gpuseed &
```

### On a hit
Without `HIT_PUBKEY`, a match is written to `seed_found.txt` (and stdout) with
the 32-hex seed, the **12-word Electrum phrase**, and the derived #71 addresses
(compressed + uncompressed). With `HIT_PUBKEY` set, the same payload is RSA-OAEP
encrypted to `hit.enc`; decrypt it on your own machine:
```bash
base64 -d hit.enc | openssl pkeyutl -decrypt -inkey hit_priv.pem -pkeyopt rsa_padding_mode:oaep
```

---

## 2. How it works

### Electrum v1 "old seed"
An Electrum v1 wallet is 128 bits of entropy with **no checksum**, so *every*
128-bit value is a valid seed. The three representations are interchangeable:

```
128-bit value  <->  32 hex chars  <->  12 words (from a 1626-word list)
```

The wallet is derived as:
1. **Stretch:** `oldseed` = the 32 ASCII hex characters. Then
   `cur = SHA256(cur || oldseed)` repeated **100 000** times. The result is the
   master private key `a`.
2. **Master public key:** `MPK = a·G` as the 64-byte uncompressed `x||y`.
3. **Child key n:** `k_n = (a + SHA256d("n:change:" + MPK)) mod N`, where
   `SHA256d` is double SHA-256 and `N` is the secp256k1 order. Addresses use the
   **uncompressed** public key (Electrum v1 default).

The 100 000-round stretch is the whole cost — it is deliberate key-stretching,
and it is why a seed search is ~100 000× slower *per candidate* than testing a
raw private key.

### The cascade (detection)
The solved puzzle keys are the low `(n-1)` bits of each child. For each candidate
master the kernel derives the child residues and compares them to the known
values — **probe keys first** (6 well-spread puzzles) for cheap rejection, then
all 83 known residues (#1–#135). A full match predicts #71 and reconstructs its
address on the CPU. No RIPEMD-160 or address hashing happens in the hot path:
the comparison is pure integer (private-key bits), so it costs ~nothing.

### GPU architecture
Two kernels per batch, intermediate data staying on the card:
1. `stretch_kernel` — generates `base + index`, formats to 32 hex, runs the
   100k-round stretch, writes the master to GPU memory. Kept lean (≈55 registers)
   so occupancy stays high.
2. `casc_kernel` — reads the master, computes `a·G` and the cascade, and on a
   match writes to a small device hit buffer via `atomicAdd`.

The host only zeros the hit counter, launches the two kernels, and reads the
counter back. There is no per-seed CPU work, which is what lets it coexist with
a CPU hunter.

### SHA-256 round
Rotations use `__funnelshift_r` (one `SHF.R.WRAP` on Ada); `ch`/`maj` use
`lop3.b32` with truth tables `0xCA` / `0xE8`. `-DPLAIN_CHMAJ` falls back to plain
C (the compiler already fuses these well — on Ada the two are equal, so this is
just an A/B switch).

---

## 3. Correctness

Everything numeric was verified on the host against independent oracles before
going into the kernel, and again on the GPU via `--selftest`:

| Component | Oracle | Result |
|---|---|---|
| `a·G` (secp256k1) | `libsecp256k1` (coincurve), 46 random + edge scalars | 46/46 |
| child `(a+z) mod N`, masked | Python + coincurve, random `a`/idx/change/bits | 28/28 |
| `lop3` truth tables `0xCA`/`0xE8` | brute vs `ch`/`maj` | 200000/200000 |
| 100k-round stretch | real Electrum v1 vector | exact |
| full chain (stretch→MPK→child) | independent from-scratch Python | bit-identical |
| mnemonic 12-word encode | round-trip hex↔words | exact |
| P2PKH address | known `privkey=1` address | exact |

---

## 4. Performance

RTX 4060 Ti (`sm_89`), measured:

| Build | seeds/s |
|---|---|
| GPU-only stretch (plain C round) | 18 559 |
| + PTX round (`lop3` + `funnelshift`) | **24 966** (+34.5%) |
| forced `lop3` vs compiler (`-DPLAIN_CHMAJ`) | identical |
| with / without `__launch_bounds__` (`-DNO_LB`) | identical |
| 2 seeds/thread ILP (`-DILP2`) | 23 831 (−4.5%) |

24 966 seeds/s ≈ **5.0 GH/s of SHA-256** at ~80% ALU utilization — the practical
ceiling for a chained INT-heavy hash on this card. The identical results across
`PLAIN_CHMAJ` / `NO_LB` show the compiler already emits optimal SASS; the ILP
regression shows the kernel is **ALU-bound, not latency-bound**. Hand-written
SASS / `.cubin` does not help SHA here — the `.cubin` is only a container, and
the speed of a well-compiled kernel is already at the hardware limit. A 5090
(~5× SHA throughput) should do ~120–130k seeds/s.

---

## 5. What we tested, and why we concluded this

Before this random searcher, the *structured* hypotheses — the ones with real
odds — were tried and all returned **zero hits**:

- **Deterministic wallets:** Electrum v1, BIP32, BIP39 (+10 wordlists), Armory.
- **Brainwallets:** rockyou × rules (~774M), plain words, indexed
  (`sha256(word+n)` and variants).
- **Structural seeds:** ~13.86M hand-built patterns (repeats, halves, mirrors,
  256-themed numbers, acrostics, …).
- **Algorithmic RNG models:** MT19937 (4 CPython-exact configurations ×
  2³² seeds), weak/small masters (31.8M), hash-counter models (several
  encodings × 2³²).

Across all of it: nothing. Combined with analysis of the solved keys
(Berlekamp–Massey, column statistics) and the long-standing community
consensus, the keys are **honestly uniform-random** — no recoverable structure.

**Why there is no shortcut for #71 itself.** Kangaroo / BSGS / Pollard-Rho all
need the *public key*, and #71 has never spent, so its pubkey is hidden. All
lattice / LLL attacks need ECDSA signatures with nonce leakage — #71 has signed
zero times. Both gateways are closed by design, so #71 can only be brute-forced
over its 2⁷⁰ key space (a job for GPU tools like keyhunt-cuda, or the community
pool), not shortcut.

**Why this seed searcher then.** If — against the evidence — the creator used an
old Electrum wallet with a weak or simply unknown seed, the KDF makes that seed
unrecoverable by any method *except* finding the seed. The structured guesses are
exhausted; this is the remaining brute-random option, offered for completeness.

---

## 6. For contributors

The most reusable part is the **cascade oracle**: a fast GPU check of "does this
wallet reproduce the solved puzzle keys?". To test a *new* seed-generation
hypothesis, replace the seed source in `stretch_kernel` (or feed masters into
`casc_kernel`) and the existing cascade validates it against all solved puzzles
at GPU speed. If you have a wallet/RNG model that wasn't in the list above, this
is the place to plug it in. Issues and PRs welcome.

---

## 7. Safety / scope

The cascade is hardcoded to the public Bitcoin Puzzle residues, so this only
recognizes the puzzle creator's wallet — it is a research/puzzle tool, not a
general wallet attacker, and the 2¹²⁸ space is not a practical threat to any
real wallet. Use it on keys you are entitled to search.

## Credits
Built by **Hristo × Claude** (Anthropic). Every crypto primitive was written from
scratch and cross-checked against `libsecp256k1` and an independent Python
implementation before shipping.

## License
MIT — see [LICENSE](LICENSE).

## Donations
BTC: bc1qg752ll0pamt9s02f3qsqceucthmmaws0g7w5r8
Thank you!
