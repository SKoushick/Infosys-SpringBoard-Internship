// TEDMind Client-side Interactivity

document.addEventListener("DOMContentLoaded", function () {
  // 1. Interactive Tag Selection Pills in Recommendation Form
  const tagPills = document.querySelectorAll(".interest-tag-pill");
  const hiddenInterestInput = document.getElementById("selectedInterestsInput");

  if (tagPills.length > 0) {
    tagPills.forEach((pill) => {
      pill.addEventListener("click", function () {
        this.classList.toggle("selected");
        this.classList.toggle("btn-outline-secondary");
        this.classList.toggle("btn-ted");

        // Collect all selected tags
        const selected = Array.from(
          document.querySelectorAll(".interest-tag-pill.selected")
        ).map((p) => p.dataset.tag);

        if (hiddenInterestInput) {
          hiddenInterestInput.value = selected.join(",");
        }
      });
    });
  }

  // 2. Real-time Talk Autocomplete Search in Recommendation Form
  const talkSearchInput = document.getElementById("talkSearchInput");
  const talkResultsList = document.getElementById("talkSearchResults");
  const seedTalkIdInput = document.getElementById("seedTalkIdInput");
  const selectedTalkDisplay = document.getElementById("selectedTalkDisplay");

  if (talkSearchInput && talkResultsList) {
    let debounceTimer;

    talkSearchInput.addEventListener("input", function () {
      clearTimeout(debounceTimer);
      const query = this.value.trim();

      if (query.length < 2) {
        talkResultsList.innerHTML = "";
        talkResultsList.classList.add("d-none");
        return;
      }

      debounceTimer = setTimeout(() => {
        fetch(`/recommendations/api/search-titles/?q=${encodeURIComponent(query)}`)
          .then((res) => res.json())
          .then((data) => {
            talkResultsList.innerHTML = "";
            if (data.results && data.results.length > 0) {
              data.results.forEach((talk) => {
                const item = document.createElement("a");
                item.href = "#";
                item.className = "list-group-item list-group-item-action py-2";
                item.innerHTML = `
                  <div class="d-flex w-100 justify-content-between align-items-center">
                    <strong class="text-dark">${escapeHtml(talk.title)}</strong>
                    <small class="text-muted"><i class="bi bi-eye"></i> ${talk.views}</small>
                  </div>
                  <small class="text-muted">${escapeHtml(talk.speaker)}</small>
                `;
                item.addEventListener("click", function (e) {
                  e.preventDefault();
                  seedTalkIdInput.value = talk.id;
                  selectedTalkDisplay.innerHTML = `
                    <div class="alert alert-success d-flex justify-content-between align-items-center py-2 px-3 mb-0">
                      <div>
                        <strong>Selected Seed Talk:</strong> ${escapeHtml(talk.title)} 
                        <span class="text-muted">(${escapeHtml(talk.speaker)})</span>
                      </div>
                      <button type="button" class="btn btn-sm btn-outline-danger" id="clearSeedTalkBtn">Change</button>
                    </div>
                  `;
                  talkResultsList.classList.add("d-none");
                  talkSearchInput.value = "";

                  document.getElementById("clearSeedTalkBtn").addEventListener("click", function () {
                    seedTalkIdInput.value = "";
                    selectedTalkDisplay.innerHTML = "";
                  });
                });
                talkResultsList.appendChild(item);
              });
              talkResultsList.classList.remove("d-none");
            } else {
              talkResultsList.innerHTML = `
                <div class="list-group-item text-muted py-2">No talks found matching "${escapeHtml(query)}"</div>
              `;
              talkResultsList.classList.remove("d-none");
            }
          })
          .catch((err) => console.error("Search error:", err));
      }, 250);
    });

    // Close autocomplete on outside click
    document.addEventListener("click", function (e) {
      if (!talkSearchInput.contains(e.target) && !talkResultsList.contains(e.target)) {
        talkResultsList.classList.add("d-none");
      }
    });
  }

  // 3. AJAX Save / Bookmark Talk buttons
  const saveButtons = document.querySelectorAll(".btn-save-talk");
  saveButtons.forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      const url = this.getAttribute("href") || this.dataset.url;
      const icon = this.querySelector("i");
      const textSpan = this.querySelector(".btn-save-text");

      fetch(url, {
        headers: { "X-Requested-With": "XMLHttpRequest" },
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.status === "ok") {
            if (data.action === "added") {
              if (icon) {
                icon.className = "bi bi-bookmark-fill text-danger";
              }
              if (textSpan) textSpan.innerText = "Saved";
              this.classList.add("btn-saved");
            } else {
              if (icon) {
                icon.className = "bi bi-bookmark";
              }
              if (textSpan) textSpan.innerText = "Save";
              this.classList.remove("btn-saved");
            }

            // Update badge if exists
            const badge = document.getElementById("savedTalksNavBadge");
            if (badge) {
              let cur = parseInt(badge.innerText) || 0;
              cur = data.action === "added" ? cur + 1 : Math.max(0, cur - 1);
              badge.innerText = cur;
              badge.style.display = cur > 0 ? "inline-block" : "none";
            }
          }
        })
        .catch((err) => console.error("Save error:", err));
    });
  });

  // 4. Auto-dismiss alerts after 4 seconds
  const alerts = document.querySelectorAll(".alert-dismissible");
  alerts.forEach((alert) => {
    setTimeout(() => {
      try {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
      } catch (e) {}
    }, 4000);
  });
});

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, function (m) {
    return map[m];
  });
}
