const revealItems = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

const galleryImage = document.querySelector("#gallery-image");
const galleryCaption = document.querySelector("#gallery-caption");
const galleryTabs = document.querySelectorAll(".gallery-tab");

if (galleryImage && galleryCaption) {
  galleryTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      galleryTabs.forEach((item) => {
        item.classList.toggle("is-active", item === tab);
        item.setAttribute("aria-pressed", String(item === tab));
      });
      galleryImage.src = tab.dataset.image;
      galleryImage.alt = tab.dataset.alt;
      galleryCaption.textContent = tab.dataset.caption;
    });
  });
}

const copyButton = document.querySelector("#copy-bibtex");
const bibtex = document.querySelector("#bibtex");
const copyStatus = document.querySelector("#copy-status");

if (copyButton && bibtex && copyStatus) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyStatus.textContent = "BibTeX copied to clipboard.";
    } catch {
      copyStatus.textContent = "Select the citation text and copy it manually.";
    }
  });
}
