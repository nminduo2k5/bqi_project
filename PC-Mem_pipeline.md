# PC-Mem: Bộ nhớ test-time cho LLM với cổng ghi có trọng số độ tự tin và pha củng cố offline

**Pipeline nghiên cứu đầy đủ và hướng dẫn triển khai**
Phiên bản 1.0 — tháng 10/2026
Mục tiêu: một bài A*/Q1 (EMNLP/ACL, hoặc TMLR / Neural Networks) chỉ cần laptop + API.

---

## Mục lục

1. Luận điểm và định vị
2. Kiến trúc tổng thể
3. Cấu trúc dữ liệu bộ nhớ
4. Cổng ghi ① (precision-weighted write gate)
5. Pha ngủ ② (offline consolidation)
6. Truy xuất và xử lý mâu thuẫn
7. Bộ prompt
8. Benchmark và bộ dữ liệu nhiễu–mâu thuẫn
9. Baseline
10. Metric
11. Thiết kế thí nghiệm và ablation
12. Ngân sách và hạ tầng
13. Cấu trúc mã nguồn
14. Lộ trình 14 tuần
15. Khung bài báo
16. Câu hỏi phản biện và câu trả lời chuẩn bị sẵn
17. Những chỗ hay hỏng
18. Checklist trước khi nộp

---

## 1. Luận điểm và định vị

### 1.1 Luận điểm trung tâm

> Các hệ bộ nhớ test-time hiện tại quyết định *ghi* dựa trên bất ngờ thô (hoặc ghi mọi thứ), nên chúng ghi nhớ nhiễu và thông tin mâu thuẫn cũng hăng hái như ghi nhớ bài học thật. Cân bất ngờ bằng **độ tự tin đã hiệu chuẩn** và tách **củng cố** thành một pha riêng khắc phục điều này mà không cần chạm vào trọng số mô hình.

Mọi phần của bài phục vụ câu này. Nếu không chứng minh được vế đầu (baseline thật sự "ô nhiễm" bộ nhớ), bài không có lý do tồn tại.

### 1.2 Khoảng trống so với tài liệu 2025–2026

| Hệ thống | Cách ghi | Củng cố | Chạm trọng số | Khoảng trống còn lại |
|---|---|---|---|---|
| RAG thuần | Ghi mọi thứ | Không | Không | Ô nhiễm bộ nhớ |
| Dynamic Cheatsheet (EACL 2026) | LLM viết lại toàn bộ bộ nhớ mỗi bước | Ngầm, không kiểm soát | Không | Một ví dụ nhiễu kéo lệch cả bộ nhớ |
| ExpRAG / ReMem (Evo-Memory, DeepMind) | Thêm kinh nghiệm, tinh chỉnh theo bước | Không có củng cố dòng/toàn cục | Không | Thừa nhận trong chính bài review |
| Titans / MIRAS (Google) | Bất ngờ = gradient | Phân rã theo thời gian | Có (MLP tự cập nhật) | Cần huấn luyện; bất ngờ thô |
| **PC-Mem (đề xuất)** | Bất ngờ × độ tự tin hiệu chuẩn + mới lạ | Pha ngủ riêng, gộp → quy tắc | Không | — |

Nguồn cảm hứng khoa học thần kinh (predictive coding: sai số có trọng số độ chính xác; hệ hồi hải mã–vỏ não: củng cố khi ngủ) được nêu **đúng một đoạn** trong Introduction. Không trích dẫn số liệu từ bài BQI.

### 1.3 Ba đóng góp tuyên bố

1. **Phân tích thực nghiệm**: bộ nhớ test-time hiện tại nhạy với nhiễu và mâu thuẫn; định nghĩa metric *pollution rate* và *downstream harm*; công bố bộ dữ liệu dòng có nhiễu–mâu thuẫn kiểm soát được.
2. **Phương pháp**: PC-Mem với cổng ghi có trọng số độ tự tin và pha củng cố offline.
3. **Kết quả**: trên Evo-Memory, MemoryBench và bộ nhiễu, kèm ablation, chuyển mô hình, và phân tích chi phí.

---

## 2. Kiến trúc tổng thể

```
Tác vụ x_t ──► Truy xuất top-k ──► LLM: dự đoán sơ bộ + độ tự tin (không nhìn bộ nhớ)
                 (semantic +           │
                  episodic)            ▼
                                 LLM: giải thật (có bộ nhớ) ──► Đáp án ──► Phản hồi (đúng/sai, reward)
                                                                              │
                                                                              ▼
                                                              ① Cổng ghi: score = a·(conf × error) + b·novelty
                                                                              │ (nếu score > θ)
                                                                              ▼
                                                                      Kho episodic (JSONL + embedding)
                                                                              │ mỗi K tác vụ
                                                                              ▼
                                                      ② Pha ngủ: gom cụm → viết quy tắc → hợp nhất → kho semantic → dọn
```

Hai luồng:

- **Online** (mỗi tác vụ): truy xuất → dự đoán → giải → phản hồi → cổng ghi.
- **Offline** (mỗi K tác vụ hoặc khi episodic đầy 80%): củng cố.

Mô hình LLM đóng băng hoàn toàn; chỉ gọi qua API. Embedding chạy local.

---

## 3. Cấu trúc dữ liệu bộ nhớ

### 3.1 Kho episodic (`episodic.jsonl`)

```json
{
  "id": "ep_0417",
  "t": 417,
  "task_id": "evo/mathqa/0417",
  "task_type": "math",
  "content": "Với phương trình bậc hai có hệ số a âm, tôi quên đổi dấu khi áp dụng công thức nghiệm → sai.",
  "trigger": "confident_error",
  "conf_pred": 0.90,
  "conf_calibrated": 0.78,
  "correct": false,
  "surprise_weighted": 0.78,
  "novelty": 0.31,
  "score": 0.68,
  "conflicts_with": [],
  "confidence": 1.0,
  "uses": 2,
  "last_used": 455,
  "consolidated": false,
  "embedding_ref": "ep_0417"
}
```

Giá trị `trigger` ∈ {`confident_error`, `novel`, `conflict`, `manual`}.

### 3.2 Kho semantic (`semantic.jsonl`)

```json
{
  "id": "sem_012",
  "rule": "Khi hệ số a < 0, nhân toàn bộ phương trình với -1 trước khi dùng công thức nghiệm.",
  "condition": "Phương trình bậc hai, a âm.",
  "support": ["ep_0417", "ep_0502", "ep_0533"],
  "n_support": 3,
  "confidence": 0.80,
  "created_t": 600,
  "updated_t": 900,
  "version": 2,
  "superseded": null
}
```

### 3.3 Chỉ mục

- Embedding: `bge-small-en-v1.5` hoặc `multilingual-e5-small` (CPU, ~100 MB).
- Lưu `numpy` array + ánh xạ id; với < 50.000 mẩu không cần FAISS.
- Mọi mẩu lưu **tại sao được ghi** (`trigger`, `score`) và **bằng chứng** (`support`) để phân tích định tính.

### 3.4 Log quyết định (`decisions.jsonl`)

Mỗi tác vụ ghi một dòng: id, retrieved ids, conf_pred, conf_calibrated, correct, novelty, score, write_decision, tokens_in, tokens_out, latency. Toàn bộ phân tích về sau đọc từ file này.

---

## 4. Cổng ghi ① (precision-weighted write gate)

### 4.1 Hai lượt gọi thay vì một

- **Lượt 1 — dự đoán** (ngắn, không bộ nhớ): đáp án sơ bộ `guess` + độ tự tin `conf_pred ∈ [0,1]`. Nếu API trả logprob, dùng logprob của token đáp án thay cho tự báo.
- **Lượt 2 — giải thật** (có bộ nhớ truy xuất): đáp án cuối.

### 4.2 Hiệu chuẩn độ tự tin

LLM tự báo độ tự tin lệch mạnh (thường quá tự tin). Quy trình:

1. 100 tác vụ đầu của dòng (giai đoạn warm-up): thu `(conf_pred, correct)`.
2. Fit **isotonic regression** `f: conf_pred → P(correct)`.
3. Từ tác vụ 101: `conf_calibrated = f(conf_pred)`; cập nhật f mỗi 200 tác vụ (rolling).
4. Báo cáo **ECE** (Expected Calibration Error) trước/sau trong bài.

Phương án không cần nhãn (ablation): dùng `conf_pred` thô, hoặc self-consistency (3 mẫu nhiệt độ 0,7, tỉ lệ đồng thuận).

### 4.3 Tính điểm và quyết định

```
error             = 1 − reward_normalized          # 1 nếu sai hoàn toàn
surprise_weighted = conf_calibrated × error         # tự tin cao mà sai = tín hiệu mạnh nhất
novelty           = 1 − max_cosine(embed(task), all_memory_embeddings)
score             = a · surprise_weighted + b · novelty        # mặc định a = 1.0, b = 0.5
write             = score > θ
```

- θ **không chọn tay**: điều chỉnh để tỉ lệ ghi ≈ 20% trên tập pilot (ngân sách ghi cố định). Mọi baseline cũng được giới hạn cùng ngân sách để so sánh công bằng.
- Trường hợp đặc biệt: `correct = true` nhưng `novelty > 0.6` → vẫn ghi với trigger `novel` (bài học "cách giải đúng cho loại bài mới").

### 4.4 Xử lý mâu thuẫn tại thời điểm ghi

Prompt ghi yêu cầu mô hình đánh dấu nếu điều cần nhớ **mâu thuẫn** với mẩu đã truy xuất:

- Không ghi đè. Tạo mẩu mới với `conflicts_with = [id_cũ]`.
- Mẩu cũ: `confidence ← confidence × 0.7`.
- Pha ngủ mới quyết định loại mẩu nào khi có ≥ 2 bằng chứng cùng chiều.

Đây là điểm khác biệt chính so với Dynamic Cheatsheet (viết lại toàn bộ, một ví dụ nhiễu kéo lệch bộ nhớ).

### 4.5 Giả mã

```python
def gate(task, truth_or_reward, retrieved, calibrator, memory, cfg):
    guess, conf_pred = llm_predict_first(task)                 # lượt 1
    answer = llm_solve(task, retrieved)                        # lượt 2
    correct, reward = evaluate(answer, truth_or_reward)
    conf_cal = calibrator.transform(conf_pred)
    error = 1 - reward
    sw = conf_cal * error
    nov = 1 - memory.max_cosine(task)
    score = cfg.a * sw + cfg.b * nov
    decision = score > cfg.theta or (correct and nov > 0.6)
    if decision:
        note, conflict_id = llm_write_note(task, guess, conf_pred, truth_or_reward, retrieved)
        memory.add_episodic(note, trigger=..., conflicts_with=conflict_id, ...)
        if conflict_id: memory.decay_confidence(conflict_id, 0.7)
    log_decision(...)
    calibrator.observe(conf_pred, correct)
    return answer
```

---

## 5. Pha ngủ ② (offline consolidation)

Chạy mỗi **K = 50** tác vụ (ablation: 25, 100, 200) hoặc khi episodic ≥ 80% dung lượng (mặc định 500 mẩu).

### 5.1 Chọn mẩu để replay

```
p_i ∝ surprise_weighted_i^0.6 × (1 + 0.1 × age_i / K)
```

Lấy tối đa 200 mẩu chưa `consolidated`. Trọng số tuổi tránh bỏ quên mẩu cũ.

### 5.2 Gom cụm

- Agglomerative clustering trên embedding, ngưỡng cosine 0,75 (ablation: 0,70 / 0,80), hoặc HDBSCAN `min_cluster_size = 3`.
- Cụm < 3 mẩu: giữ nguyên trong episodic.

### 5.3 Viết quy tắc

Một lượt API cho mỗi cụm (prompt §7.4). Đầu ra: `rule` (≤ 50 từ), `condition` (≤ 20 từ), hoặc `KHÔNG GỘP` nếu mâu thuẫn không giải được.

### 5.4 Hợp nhất vào semantic

- Nếu quy tắc mới có cosine > 0,85 với quy tắc cũ: gọi một lượt hợp nhất (prompt §7.5), tăng `version`, cộng `support`.
- Nếu không: thêm mới.
- Đây là phiên bản text của `M ← (1−δ)M + δ·M_new`.

### 5.5 Giải quyết mâu thuẫn

Với mỗi cặp `conflicts_with`: đếm bằng chứng mỗi phía (số mẩu episodic ủng hộ). Phía có ≥ 2 bằng chứng và nhiều hơn phía kia thắng; phía thua `superseded = id_thắng`, loại khỏi truy xuất.

### 5.6 Dọn

- Mẩu đã gộp: `consolidated = true`, loại khỏi truy xuất mặc định, giữ file.
- Mẩu `uses = 0` sau 200 tác vụ và `surprise_weighted < 0.2`: xoá.
- Semantic: nếu `confidence < 0.3` sau 3 pha ngủ: xoá.

### 5.7 Giả mã

```python
def sleep(memory, embedder, llm, cfg):
    cand = memory.sample_episodic(weights=priority, n=200)
    clusters = cluster(embedder(cand), threshold=cfg.cos_cluster)
    for c in clusters:
        if len(c) < 3: continue
        rule = llm_write_rule(c)                  # có thể trả KHÔNG GỘP
        if rule is None: continue
        near = memory.nearest_semantic(rule, cfg.cos_merge)
        if near: memory.merge_semantic(near, rule, support=c)
        else:    memory.add_semantic(rule, support=c)
        memory.mark_consolidated(c)
    memory.resolve_conflicts(min_evidence=2)
    memory.prune(cfg)
```

---

## 6. Truy xuất và xử lý mâu thuẫn

- Top-3 semantic (loại `superseded`, `confidence < 0.3`) + top-3 episodic chưa gộp, theo cosine với embedding tác vụ.
- Nếu hai mẩu truy xuất mâu thuẫn nhau: đưa cả hai vào prompt với nhãn `[A]`, `[B — mâu thuẫn với A]`; để mô hình quyết; log để phân tích.
- Mỗi lần truy xuất: `uses += 1`, `last_used = t`.
- Ngân sách token cho phần bộ nhớ trong prompt: cố định (ví dụ 600 token) cho mọi hệ thống.

---

## 7. Bộ prompt

Giữ ngắn, nhiệt độ 0 cho giải và ghi; 0,3 cho viết quy tắc.

### 7.1 Dự đoán sơ bộ (lượt 1)

```
Bạn sẽ được giao một bài. CHƯA giải chi tiết. Trả lời đúng 2 dòng:
ANSWER_GUESS: <đáp án dự đoán nhanh nhất>
CONFIDENCE: <số thực 0–1, xác suất bạn đúng>
Bài: {task}
```

### 7.2 Giải thật (lượt 2)

```
Ghi chú từ kinh nghiệm trước (có thể liên quan, có thể không):
{retrieved_semantic}
{retrieved_episodic}
Nếu hai ghi chú mâu thuẫn, hãy tự quyết dựa trên bài hiện tại.

Bài: {task}
{format_instructions}
```

### 7.3 Ghi mẩu episodic (chỉ khi cổng mở)

```
Bài: {task}
Dự đoán ban đầu: {guess} (tự tin {conf_pred})
Kết quả đúng / phản hồi: {truth_or_feedback}
Ghi chú liên quan hiện có: {retrieved}

Viết MỘT câu, tối đa 40 từ, ghi lại điều cần nhớ để lần sau không sai (hoặc cách giải đúng cho loại bài mới).
Nếu điều này MÂU THUẪN với một ghi chú hiện có, bắt đầu bằng: "MÂU THUẪN với <id>:".
```

### 7.4 Viết quy tắc (pha ngủ)

```
Dưới đây là {n} ghi chú từ các bài khác nhau:
{notes_with_ids}

Viết MỘT quy tắc tổng quát (≤ 50 từ) giải thích chung cho các ghi chú này, và điều kiện áp dụng (≤ 20 từ).
Định dạng:
RULE: ...
CONDITION: ...
Nếu các ghi chú mâu thuẫn và không thể gộp, trả lời đúng: KHÔNG GỘP
```

### 7.5 Hợp nhất quy tắc

```
Quy tắc cũ (phiên bản {v}): {old_rule} | Điều kiện: {old_cond}
Quy tắc mới: {new_rule} | Điều kiện: {new_cond}
Viết một quy tắc hợp nhất (≤ 60 từ) và điều kiện (≤ 25 từ) bao phủ cả hai. Nếu chúng loại trừ nhau, trả lời: GIỮ RIÊNG
```

---

## 8. Benchmark và bộ dữ liệu nhiễu–mâu thuẫn

### 8.1 Benchmark có sẵn

| Benchmark | Vai trò | Lý do |
|---|---|---|
| **Evo-Memory** (DeepMind, 2026) | Chính | 10 dataset dạng dòng, có sẵn baseline ExpRAG/ReMem, cộng đồng biết |
| **MemoryBench** (ICML 2026) | Phụ | Đo học liên tục của LLMsys, có baseline RAG |
| ARC-AGI-style / BBH streams | Tuỳ chọn | Kiểm tra tổng quát hoá từ quy tắc |

Dùng đúng split và protocol của họ; không sửa.

### 8.2 Bộ nhiễu–mâu thuẫn tự tạo (`noisy_stream`) — đóng góp của bài

Lấy một dòng tác vụ sạch (ví dụ 2 dataset của Evo-Memory, 1.000 tác vụ mỗi dòng), chèn:

- **Nhiễu (15%)**: tác vụ vô nghĩa (token xáo trộn, câu không có đáp án), phản hồi ngẫu nhiên.
- **Mâu thuẫn (15%)**: tác vụ hợp lệ nhưng phản hồi "đúng" bị **cố ý làm sai** theo một quy luật ổn định (ví dụ luôn lệch một bước) để có thể đo xem bộ nhớ có học sai không.
- **Lặp (10%)**: tác vụ cũ xuất hiện lại sau 100–300 bước để đo forgetting.

Quy tắc bắt buộc:

- Tạo bộ dữ liệu **trước khi** chỉnh θ, K, k. Seed cố định, mã sinh công bố.
- Đánh nhãn ẩn từng tác vụ (`clean / noise / conflict / repeat`) — không lộ cho hệ thống.
- Tạo 3 phiên bản với tỉ lệ nhiễu 5 / 15 / 30% để vẽ đường cong độ nhạy.

---

## 9. Baseline

Tất cả chạy cùng mô hình nền, cùng ngân sách token bộ nhớ trong prompt, cùng ngân sách ghi (≈ 20%) khi áp dụng được.

1. **No-memory**: giải trực tiếp.
2. **RAG-all**: ghi mọi tác vụ + đáp án; truy xuất top-k.
3. **RAG-budget**: như trên nhưng chỉ ghi 20% ngẫu nhiên (kiểm soát ngân sách).
4. **Dynamic Cheatsheet**: mã gốc của tác giả.
5. **ExpRAG** và **ReMem**: mã Evo-Memory.
6. **PC-Mem (raw surprise)**: score = error + b·novelty, bỏ conf → ablation quan trọng nhất.
7. **PC-Mem (no sleep)**: cổng ① nhưng không củng cố.
8. **PC-Mem (full)**.

---

## 10. Metric

| Metric | Định nghĩa | Trả lời câu hỏi |
|---|---|---|
| Acc@time | Độ chính xác tích luỹ theo t (đường cong học) | Có học không |
| Acc-last-20% | Độ chính xác trên 20% tác vụ cuối | Hiệu quả sau khi tích luỹ |
| Forgetting | Acc trên tác vụ lặp − Acc lần đầu | Có quên không |
| **Pollution rate** | % mẩu trong bộ nhớ bắt nguồn từ tác vụ `noise`/`conflict` | Bộ nhớ sạch không |
| **Downstream harm** | Δ Acc trên tác vụ sạch khi bộ nhớ chứa mẩu bẩn vs. bộ nhớ lọc sạch (counterfactual) | Mẩu bẩn có gây hại không |
| Generalization | Acc trên tác vụ mới cùng quy luật với cụm đã gộp, trước vs. sau pha ngủ | Củng cố tạo tổng quát hoá không |
| Memory size | Số mẩu và tổng token theo t | Có phình không |
| ECE | Trước/sau hiệu chuẩn | Hiệu chuẩn có tác dụng |
| Cost | Token vào/ra, số lượt gọi, độ trễ | Giá phải trả |

Mọi metric: trung bình ± khoảng tin cậy 95% qua 3 seed (thứ tự dòng). Kiểm định: paired bootstrap trên tác vụ.

---

## 11. Thiết kế thí nghiệm và ablation

### 11.1 Thí nghiệm then chốt (làm đầu tiên — Figure 1)

Trên `noisy_stream` 15%: pollution rate và downstream harm của RAG-all, Dynamic Cheatsheet, PC-Mem (raw), PC-Mem (full). Kỳ vọng: raw surprise ghi nhiễu nhiều (nhiễu luôn bất ngờ); PC-Mem ghi ít hơn hẳn. Nếu kết quả này không ra, dừng và xem lại cổng trước khi làm gì khác.

### 11.2 Kết quả chính

Evo-Memory (toàn bộ) + MemoryBench: Acc@time, Acc-last-20%, forgetting, size, cost cho 8 hệ thống.

### 11.3 Ablation

- Bỏ hiệu chuẩn (dùng conf thô); bỏ conf (raw surprise); bỏ novelty.
- Bỏ pha ngủ; replay ngẫu nhiên thay ưu tiên; không hợp nhất (chỉ chuyển nguyên mẩu sang semantic).
- K ∈ {25, 50, 100, 200}; ngưỡng cụm ∈ {0,70, 0,75, 0,80}; ngân sách ghi ∈ {10, 20, 40%}.

### 11.4 Chuyển mô hình

Chạy lại kết quả chính với một mô hình mở (Qwen 7B hoặc 32B qua Together/Groq) để chứng minh không phụ thuộc nhà cung cấp.

### 11.5 Độ nhạy nhiễu

Đường cong Acc-last-20% theo tỉ lệ nhiễu 5 / 15 / 30%.

### 11.6 Phân tích định tính

5–10 ví dụ thật: mẩu episodic → quy tắc semantic; trường hợp cổng từ chối ghi thứ đáng ghi; trường hợp mâu thuẫn được giải đúng/sai.

---

## 12. Ngân sách và hạ tầng

### 12.1 Ước lượng token

- Evo-Memory ≈ 5.000–8.000 tác vụ; 8 cấu hình; 3 seed; ~3 lượt gọi/tác vụ; ~1.500 token/lượt → ≈ 400–500 triệu token.
- Mô hình: GPT-4o-mini hoặc Gemini 2.5 Flash cho chính (≈ 150–250 USD tổng); Qwen qua Groq/Together cho chuyển mô hình (rẻ hơn).
- **Pilot** 500 tác vụ × 3 cấu hình trước (≈ 10 USD) để chỉnh θ, K, k.

### 12.2 Bắt buộc

- **Cache mọi lời gọi API** vào SQLite theo hash(prompt, model, temperature). Debug không cache tốn gấp 3.
- Giới hạn chi tiêu theo ngày trong tài khoản API.
- Embedding local (CPU đủ).

### 12.3 Laptop

Mọi tính toán local (embedding, clustering, phân tích) chạy được trên laptop 8–16 GB RAM. Không cần GPU.

---

## 13. Cấu trúc mã nguồn

```
pcmem/
  config.py          # mọi tham số: a, b, theta, K, k, ngưỡng cosine, ngân sách
  llm.py             # wrapper OpenAI-compatible; retry; cache SQLite; đếm token
  embed.py           # bge-small / e5-small local; cosine utils
  memory.py          # Episodic, Semantic, add/retrieve/decay/prune; conflict check
  calibrate.py       # isotonic rolling; ECE
  gate.py            # predict-first, score, decide, write note
  sleep.py           # sample, cluster, write rule, merge, resolve conflicts, prune
  runner.py          # chạy một dòng tác vụ; log decisions.jsonl; checkpoint
  streams/
    evo_memory.py    # loader + evaluator theo protocol gốc
    memorybench.py
    noisy_stream.py  # sinh bộ nhiễu–mâu thuẫn, nhãn ẩn, seed
  baselines/
    rag.py, cheatsheet_adapter.py, exprag_adapter.py
  analysis/
    curves.py        # Acc@time, forgetting
    pollution.py     # pollution rate, downstream harm (counterfactual)
    cost.py
    qualitative.py   # trích ví dụ
tests/               # unit test cho gate (score), memory (conflict), sleep (merge)
scripts/
  run_pilot.sh, run_full.sh, make_noisy.sh, make_figures.sh
```

Nguyên tắc: mọi quyết định của hệ thống phải tái tạo được từ `decisions.jsonl` + `episodic.jsonl` + `semantic.jsonl`.

---

## 14. Lộ trình 14 tuần

| Tuần | Việc | Đầu ra kiểm tra được |
|---|---|---|
| 1 | `llm.py` + cache, `embed.py`, loader Evo-Memory; chạy No-memory và RAG-all trên 500 tác vụ | Số baseline khớp bài gốc trong sai số hợp lý |
| 2 | **Tạo `noisy_stream` (trước khi chỉnh tham số)**; `memory.py` | Bộ dữ liệu 3 mức nhiễu, seed cố định, công bố mã sinh |
| 3 | `calibrate.py`, `gate.py`; pilot 500 tác vụ; đọc log xem ghi *gì* | ECE giảm; ví dụ mẩu hợp lý |
| 4 | Thí nghiệm then chốt (§11.1) | **Figure 1** — nếu không ra, quay lại tuần 3 |
| 5–6 | `sleep.py`; kiểm tra size không tăng tuyến tính, forgetting giảm | Generalization trước/sau ngủ |
| 7 | Adapter Dynamic Cheatsheet, ExpRAG, ReMem | Chạy được trên cùng dòng |
| 8–10 | Chạy đầy đủ 3 seed, 8 cấu hình, 2 benchmark; chuyển mô hình | Bảng kết quả chính + ablation |
| 11 | Phân tích: pollution, harm, cost, định tính | Mọi hình trong bài |
| 12–13 | Viết bản thảo; nội bộ phản biện | Bản thảo hoàn chỉnh |
| 14 | Chỉnh sửa, nộp ARR/EMNLP; đẩy arXiv cùng ngày | Nộp |

---

## 15. Khung bài báo

**Tên (mẫu)**: *Not All Surprises Are Worth Remembering: Precision-Weighted Writes and Offline Consolidation for Test-Time LLM Memory*

| Phần | Độ dài | Nội dung bắt buộc |
|---|---|---|
| Abstract | 150 từ | Vấn đề → con số về ô nhiễm của baseline → hai cơ chế → kết quả có số → mã mở |
| 1. Introduction | 1 trang | Ví dụ agent nhớ nhầm; khoảng trống với trích dẫn chính xác; 3 đóng góp; **một đoạn** về cảm hứng thần kinh |
| 2. Related Work | 0,75 trang | Bộ nhớ test-time; hiệu chuẩn LLM; continual learning/consolidation — mỗi đoạn kết bằng câu "khác gì" |
| 3. Memory Pollution | 0,75 trang | Định nghĩa hình thức; metric; bộ dữ liệu; **Figure 1** với baseline |
| 4. Method | 1,5 trang | Cổng ①, hiệu chuẩn, mâu thuẫn; Algorithm 1 pha ngủ; hình kiến trúc; nêu rõ chi phí thêm |
| 5. Experiments | 2,5 trang | Thiết lập (ngân sách công bằng); kết quả chính; nhiễu; ablation; chuyển mô hình; chi phí |
| 6. Analysis | 1 trang | Ghi gì, gộp gì, lỗi gì; reliability diagram |
| 7. Limitations | 0,5 trang | Tự tin tự báo; cần phản hồi; +1 lượt gọi; quy mô |
| Appendix | — | Prompt đầy đủ, tham số, bộ dữ liệu, ví dụ thêm |

**Hình tối thiểu**: F1 pollution rate (bar + CI); F2 kiến trúc; F3 đường cong học; F4 reliability + tỉ lệ ghi theo bin tự tin. **Bảng**: T1 kết quả chính; T2 ablation; T3 chi phí.

**Tuyên bố nên dùng**: "giảm pollution rate từ X% xuống Y% với cùng ngân sách ghi"; "Acc-last-20% tăng Δ, p < 0,05 qua 3 seed".
**Tránh**: "mô phỏng não người"; "giải quyết catastrophic forgetting"; "SOTA" nếu chỉ thắng trên bộ tự tạo.

**Nơi nộp**: ARR → EMNLP/ACL/NAACL (A*). Dự phòng: TMLR, Neural Networks (Q1). Đẩy arXiv ngay khi nộp.

---

## 16. Câu hỏi phản biện và câu trả lời chuẩn bị sẵn

| Câu hỏi | Trả lời |
|---|---|
| Khác Titans/MIRAS? | Họ cập nhật trọng số, cần huấn luyện, bất ngờ thô; chúng tôi ở tầng text, không huấn luyện, cân bằng độ tự tin hiệu chuẩn, có pha củng cố riêng. |
| Bộ nhiễu do bạn tạo, thiên vị? | Tạo trước khi chỉnh tham số (có commit log), mã sinh công bố, 3 mức nhiễu, và cũng thắng trên Evo-Memory gốc. |
| Hiệu chuẩn cần nhãn? | Chỉ 100 tác vụ warm-up có phản hồi; báo cáo cả phiên bản không hiệu chuẩn và self-consistency. |
| Thêm lượt gọi thì so sánh công bằng không? | Baseline được cùng ngân sách token (tăng k truy xuất); bảng chi phí minh bạch. |
| Chỉ một mô hình nền? | Chuyển sang mô hình mở; kết quả giữ chiều. |
| Khác Dynamic Cheatsheet? | DC viết lại toàn bộ bộ nhớ mỗi bước, không có cổng, không có bằng chứng; chúng tôi ghi chọn lọc, lưu bằng chứng, giải mâu thuẫn bằng đa số. |
| Phần thần kinh học có cần không? | Chỉ là nguồn ý tưởng, một đoạn; mọi tuyên bố đứng bằng benchmark. |

---

## 17. Những chỗ hay hỏng

1. **Không cache API** → tốn gấp 3, không tái tạo được.
2. **Tạo bộ nhiễu sau khi chỉnh θ** → tối ưu trên test; phản biện bắt.
3. **Dùng conf thô** → cổng gần như ngẫu nhiên; phải hiệu chuẩn và báo cáo ECE.
4. **Baseline không cùng ngân sách** → mọi so sánh vô nghĩa.
5. **Trộn seed/thứ tự dòng** → kết quả dòng rất nhạy với thứ tự; 3 seed là tối thiểu.
6. **Pha ngủ quá hăng** (ngưỡng cụm thấp) → gộp sai, quy tắc mơ hồ; kiểm tra bằng Generalization metric.
7. **Prompt dài dần** → chi phí tăng ngầm; cố định ngân sách token bộ nhớ.
8. **Không đọc log** → bỏ sót lỗi hệ thống ghi toàn thứ vô nghĩa; đọc 50 mẩu ngẫu nhiên mỗi pilot.

---

## 18. Checklist trước khi nộp

- [ ] Bộ nhiễu–mâu thuẫn tạo trước khi chỉnh tham số (commit log chứng minh)
- [ ] 3 seed, CI 95%, kiểm định paired bootstrap
- [ ] Mọi baseline cùng mô hình nền, cùng ngân sách token và ngân sách ghi
- [ ] ECE trước/sau hiệu chuẩn báo cáo
- [ ] Bảng chi phí (token, lượt gọi, độ trễ)
- [ ] Chuyển mô hình mở
- [ ] Ablation: raw surprise, no sleep, no calibration, random replay
- [ ] Phân tích định tính với ví dụ thật
- [ ] Mã, prompt, dữ liệu, log quyết định công bố (GitHub + Zenodo DOI)
- [ ] Limitations trung thực
- [ ] Không có tuyên bố "não người", "SOTA" vô căn cứ
- [ ] arXiv cùng ngày nộp
