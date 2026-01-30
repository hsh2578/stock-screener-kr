# UX 개선 사항 가이드

## 📋 적용 필요한 개선 사항

이 문서는 `주식_스크리너_전체.html` 파일에 적용해야 할 UX 개선 코드를 정리한 것입니다.

---

## 1. 🔴 색상만으로 정보 전달 개선 (WCAG 1.4.1 Level A)

### 문제
상승/하락을 색상만으로 구분하여 색맹 사용자가 인지 불가

### 해결 방법
등락률 앞에 아이콘 추가

#### CSS 추가 (line ~456 이후)
```css
/* 상승/하락 아이콘 - 색맹 접근성 */
.positive::before {
    content: '▲ ';
    font-size: 0.85em;
}
.negative::before {
    content: '▼ ';
    font-size: 0.85em;
}
```

---

## 2. 🔴 키보드 네비게이션 완성 (WCAG 2.1.1 Level A)

### 문제
- 테이블 행이 onClick만 있어 키보드로 접근 불가
- 필터 버튼 키보드 이벤트 미처리
- 모달 포커스 트랩 미구현

### 해결 방법 A: 테이블 행 키보드 접근성

#### JavaScript 수정 (line ~2520 renderTable 함수 내부)
```javascript
// 기존 코드
row.onclick = () => openChartModal(item);

// 개선 코드로 변경
row.onclick = () => openChartModal(item);
row.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openChartModal(item);
    }
};
row.setAttribute('tabindex', '0');
row.setAttribute('role', 'button');
```

### 해결 방법 B: 필터 버튼 키보드 지원

#### JavaScript 수정 (line ~3070 toggleMA150Filter 함수)
```javascript
function toggleMA150Filter(screenerId) {
    const btn = document.getElementById(screenerId ?
        `filter-ma150-${screenerId}` :
        'filter-ma150-btn');

    if (!btn) return;

    const isActive = btn.classList.contains('active');
    btn.classList.toggle('active');
    btn.setAttribute('aria-pressed', !isActive); // 이미 있음 - 확인

    // 필터링 로직...

    // 키보드 이벤트 추가
    btn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            btn.click();
        }
    });
}
```

### 해결 방법 C: 모달 포커스 트랩

#### JavaScript 추가 (line ~2311 openChartModal 함수 수정)
```javascript
async function openChartModal(stock) {
    currentStock = stock;
    modalOverlay.classList.add('active');

    // 기존 모달 오픈 로직...

    // 포커스 트랩 추가
    const modalContent = document.querySelector('.modal-content');
    const focusableElements = modalContent.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    // 첫 번째 요소에 포커스
    firstFocusable.focus();

    // Tab 키 트랩
    const trapFocus = (e) => {
        if (e.key !== 'Tab') return;

        if (e.shiftKey) {
            if (document.activeElement === firstFocusable) {
                e.preventDefault();
                lastFocusable.focus();
            }
        } else {
            if (document.activeElement === lastFocusable) {
                e.preventDefault();
                firstFocusable.focus();
            }
        }
    };

    modalContent.addEventListener('keydown', trapFocus);

    // 모달 닫을 때 이벤트 제거
    modalOverlay.addEventListener('click', () => {
        modalContent.removeEventListener('keydown', trapFocus);
    }, { once: true });
}
```

---

## 3. 🟡 모바일 터치 타겟 크기 개선

### 문제
테이블 정렬 버튼이 44px 미만

### 해결 방법

#### CSS 수정 (line ~392)
```css
/* 기존 코드 */
th {
    background-color: var(--color-bg-tertiary);
    padding: 14px 14px;
    /* ... */
}

/* 개선 코드로 변경 */
th {
    background-color: var(--color-bg-tertiary);
    padding: 15px 14px; /* 높이 증가 */
    min-height: 48px; /* 최소 높이 보장 */
    /* ... */
}

@media (max-width: 768px) {
    th {
        padding: 17px 8px; /* 모바일에서 더 큰 터치 영역 */
        min-height: 52px;
    }
}
```

---

## 4. 🟡 에러 메시지 개선

### 문제
네트워크 오류 시 구체적인 안내 부족

### 해결 방법

#### JavaScript 수정 (line ~1864 loadAllData 함수)
```javascript
// 기존 catch 블록
} catch (e) {
    console.warn(`  ⚠ ${key}: 로드 실패 - ${e.message}`);
}

// 개선 코드로 변경
} catch (e) {
    console.warn(`  ⚠ ${key}: 로드 실패 - ${e.message}`);

    // 사용자에게 알림
    if (e.message.includes('Failed to fetch')) {
        showToast(
            '네트워크 오류',
            `${filename} 파일을 불러올 수 없습니다. 인터넷 연결을 확인해주세요.`,
            5000
        );
    }
}
```

---

## 5. 🟢 테이블 정렬 상태 명확성 개선

### 문제
정렬 아이콘 ⇅가 작고 현재 정렬 방향 불명확

### 해결 방법

#### JavaScript 수정 (line ~2470 sortTable 함수)
```javascript
function sortTable(key, screenerId) {
    const state = tableStates[screenerId];

    // 정렬 방향 전환
    if (state.sortKey === key) {
        state.sortOrder = state.sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortKey = key;
        state.sortOrder = 'desc';
    }

    // 기존 정렬 로직...

    // 헤더 업데이트 (ARIA 속성 추가)
    const headers = document.querySelectorAll(`#${screenerId}-table th`);
    headers.forEach(th => {
        th.classList.remove('sorted');
        th.removeAttribute('aria-sort');
        const icon = th.querySelector('.sort-icon');
        if (icon) icon.textContent = '⇅';
    });

    const currentHeader = Array.from(headers).find(th =>
        th.textContent.includes(getColumnName(key))
    );

    if (currentHeader) {
        currentHeader.classList.add('sorted');
        currentHeader.setAttribute('aria-sort',
            state.sortOrder === 'asc' ? 'ascending' : 'descending'
        );

        const icon = currentHeader.querySelector('.sort-icon');
        if (icon) {
            icon.textContent = state.sortOrder === 'asc' ? '▲' : '▼';
        }
    }
}
```

---

## 6. 🟢 필터 적용 후 Toast 알림 추가

### 문제
필터 적용 시 시각적 피드백 부족

### 해결 방법

#### JavaScript 수정 (line ~3070 toggleMA150Filter 함수 마지막)
```javascript
function toggleMA150Filter(screenerId) {
    // 기존 필터링 로직...

    // Toast 알림 추가
    const filteredCount = filteredData.length;
    const totalCount = data.data.length;

    if (isActive) {
        showToast(
            '필터 적용됨',
            `150일선 위 종목 ${filteredCount}개로 필터링했습니다.`,
            3000
        );
    } else {
        showToast(
            '필터 해제됨',
            `전체 ${totalCount}개 종목을 표시합니다.`,
            3000
        );
    }
}
```

---

## 7. 🟢 Breadcrumb 구조적 마크업 개선

### 문제
Breadcrumb이 시각적으로만 존재

### 해결 방법

#### HTML 수정 (각 페이지의 breadcrumb 섹션, 예: line ~1279)
```html
<!-- 기존 코드 -->
<nav class="breadcrumb" aria-label="breadcrumb">
    <a href="#" onclick="showPage('home'); return false;">홈</a>
    <span class="separator" aria-hidden="true">›</span>
    <span class="current">박스권 스크리너</span>
</nav>

<!-- 개선 코드로 변경 -->
<nav class="breadcrumb" aria-label="breadcrumb">
    <ol style="display:flex;align-items:center;gap:8px;list-style:none;margin:0;padding:0;">
        <li>
            <a href="#" onclick="showPage('home'); return false;">홈</a>
        </li>
        <li aria-hidden="true" style="color:var(--color-border-dark);">›</li>
        <li aria-current="page" style="color:var(--color-text-primary);font-weight:600;">
            박스권 스크리너
        </li>
    </ol>
</nav>
```

---

## 8. 🟢 데이터 업데이트 시간 상대 표시

### 문제
날짜만 표시되어 신선도 파악 어려움

### 해결 방법

#### JavaScript 추가 (유틸리티 함수)
```javascript
function formatRelativeDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return '방금 전';
    if (diffMins < 60) return `${diffMins}분 전`;
    if (diffHours < 24) return `${diffHours}시간 전`;
    if (diffDays === 0) return '오늘';
    if (diffDays === 1) return '어제';
    if (diffDays < 7) return `${diffDays}일 전`;

    return dateString; // 일주일 이상은 그대로 표시
}
```

#### JavaScript 수정 (line ~2650 renderPage 함수에서 날짜 표시 부분)
```javascript
// 기존 코드
dateSpan.textContent = `데이터 기준: ${data.date}`;

// 개선 코드로 변경
const relativeDate = formatRelativeDate(data.date);
dateSpan.textContent = `데이터 기준: ${data.date}`;
dateSpan.title = `${relativeDate} 업데이트`; // 툴팁으로 상대 시간 표시
```

---

## 9. 🟢 모달 사용성 개선

### 문제
모달 닫기 방법 안내 부족

### 해결 방법

#### HTML 수정 (line ~1745 모달 헤더)
```html
<div class="modal-header">
    <div class="modal-stock-info">
        <h2>
            <span class="stock-name"></span>
            <span class="ticker"></span>
        </h2>
        <div class="price-row">
            <span class="price"></span>
            <span class="change"></span>
        </div>
        <!-- 닫기 안내 추가 -->
        <div style="margin-top:8px;font-size:12px;color:var(--color-text-muted);">
            <span aria-hidden="true">💡</span>
            차트를 클릭하거나 <kbd style="padding:2px 6px;background:var(--color-bg-tertiary);border-radius:4px;font-family:monospace;">ESC</kbd>로 닫기
        </div>
    </div>
    <button class="modal-close" onclick="closeChartModal()" aria-label="차트 닫기">×</button>
</div>
```

---

## ✅ 적용 우선순위

### 즉시 적용 (Critical)
1. **색상 + 아이콘 조합** (개선 1)
2. **키보드 네비게이션** (개선 2)
3. **모바일 터치 타겟** (개선 3)

### 단기 적용 (1-2주 내)
4. **에러 메시지 개선** (개선 4)
5. **정렬 상태 ARIA** (개선 5)
6. **필터 Toast 알림** (개선 6)

### 중장기 적용 (향후 업데이트)
7. **Breadcrumb 구조화** (개선 7)
8. **상대 시간 표시** (개선 8)
9. **모달 안내 추가** (개선 9)

---

## 🧪 테스트 체크리스트

개선 사항 적용 후 다음을 확인하세요:

### 접근성 테스트
- [ ] 키보드만으로 모든 기능 사용 가능 (Tab, Enter, Space, ESC)
- [ ] 스크린리더로 테이블 정렬 상태 읽기 가능
- [ ] 색맹 시뮬레이터로 등락 구분 가능
- [ ] 모달 포커스 트랩 작동 확인

### 모바일 테스트
- [ ] 모든 버튼 터치 영역 44×44px 이상
- [ ] 카드뷰 터치 피드백 명확
- [ ] 필터 버튼 클릭 쉬움

### 브라우저 테스트
- [ ] Chrome, Safari, Firefox 모두 테스트
- [ ] iOS Safari, Android Chrome 테스트
- [ ] 다크모드 지원 (향후)

---

## 📚 참고 자료

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [iOS Human Interface Guidelines - Touch Targets](https://developer.apple.com/design/human-interface-guidelines/ios/user-interaction/touch-targets/)
- [Material Design - Accessibility](https://material.io/design/usability/accessibility.html)
- [네이버 금융 UI 패턴](https://finance.naver.com/)
