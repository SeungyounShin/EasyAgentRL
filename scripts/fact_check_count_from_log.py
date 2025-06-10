import re
from collections import Counter

log_file_path = './logs/searchR1-like2025-06-01-12-46.log'
per_epoch = 414

REFLECTIVE_PHRASES = [
    r"think again",
    r"let'?s verify",
    r"verify",
    r"fact ?check",
    r"double[- ]?check",
    r"cross[- ]?check",
    r"re[- ]?evaluate",
    r"search with (?:a )?different query",
    r"search with another query",
    r"try (?:another|a different) search",
    r"use (?:another|a different) query",
    r"re[- ]?search",                    # ‘re-search’ ‘research’ 혼동 방지: 필요하면 \bre[- ]?search\b
    r"look (?:this|it) up",
    r"let'?s look (?:this|it) up",
    r"look .* on the web",
]

def extract_score_to_prompt_sections(log_file_path):
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            log_content = file.read()
        
        # [score]로 끝나고 [prompt]로 시작하는 패턴 찾기
        pattern = r'\[score\].*?\n(.*?)\[prompt\]'
        matches = re.findall(pattern, log_content, re.DOTALL)
        
        print(f"총 {len(matches)}개의 [score] -> [prompt] 섹션을 찾았습니다.\n")
        
        # for i, match in enumerate(matches, 1):
        #     print(f"=== 섹션 {i} ===")
        #     print(match.strip())
        #     print("-" * 50)
            
        return matches
        
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {log_file_path}")
        return []
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")
        return []

def count_reflective_phrases_per_epoch(
    sections,
    per_epoch: int = 414,
    phrases: list[str] | None = None,
    ignore_case: bool = True,
):
    """
    sections : extract_score_to_prompt_sections()로 얻은 리스트
    per_epoch: 한 에포크에 포함되는 섹션 수
    phrases  : 집계할 패턴 목록(정규식). None이면 기본 REFLECTIVE_PHRASES 사용
    ignore_case: 대/소문자 구분 여부
    반환값   : [
                 {"epoch": 0, "<phrase1>": n, "<phrase2>": m, ..., "TOTAL": t},
                 {"epoch": 1, ...},
                 ...
               ]
    """
    if phrases is None:
        phrases = REFLECTIVE_PHRASES

    # 컴파일해 두면 약간 더 빠름
    flags = re.IGNORECASE if ignore_case else 0
    compiled = {p: re.compile(p, flags) for p in phrases}

    results = []
    for ei, start in enumerate(range(0, len(sections), per_epoch)):
        epoch_sections = sections[start:start + per_epoch]
        epoch_counter = Counter()

        for sec in epoch_sections:
            for p, rex in compiled.items():
                # 한 섹션에서 몇 번이든 모두 세기
                epoch_counter[p] += len(rex.findall(sec))

        epoch_counter["TOTAL"] = sum(epoch_counter.values())
        epoch_counter["epoch"] = ei
        results.append(dict(epoch_counter))

    return results
# 함수 실행 
sections = extract_score_to_prompt_sections(log_file_path)
print(f"총 섹션 수: {len(sections)}")
print(f"총 섹션 수: {len(sections) / per_epoch}")

# 사용 예시
epoch_stats = count_reflective_phrases_per_epoch(sections, per_epoch=414)
for row in epoch_stats:
    epoch = row.pop("epoch")
    total = row.pop("TOTAL")
    # print(f"[Epoch {epoch}] TOTAL={total} │ " + " │ ".join(f"{k}: {v}" for k, v in row.items()))
    # print total only
    print(f"[Epoch {epoch}] TOTAL={total}")
