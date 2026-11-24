"""
End-to-End Playwright test suite for AquaSense AI.
Verifies all pages, interactive states, theme toggle, form validation,
batch CSV upload, results tabs, diagnostic lightbox, accordions, and mobile responsiveness.
"""

import urllib.request

import pytest

try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    Page = None  # type: ignore[assignment, misc]
    expect = None  # type: ignore[assignment]


def _check_server_active() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


SERVER_ACTIVE = _check_server_active()

pytestmark = pytest.mark.skipif(
    not (HAS_PLAYWRIGHT and SERVER_ACTIVE),
    reason="Playwright E2E tests require playwright and an active server at http://127.0.0.1:8000",
)

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(autouse=True)
def setup_console_error_tracker(page: Page):
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    yield
    # Assert zero unhandled runtime JavaScript exceptions
    assert len(errors) == 0, f"Page threw unhandled JS exceptions: {errors}"


def test_homepage_loads_and_has_disclaimer(page: Page):
    page.goto(f"{BASE_URL}/")
    expect(page).to_have_title("AquaSense AI: Environmental Water-Quality Intelligence Platform")
    
    # Persistent statutory disclaimer banner
    disclaimer = page.locator(".disclaimer-banner")
    expect(disclaimer).to_be_visible()
    expect(disclaimer).to_contain_text("Screening support only")
    
    # Two-column hero elements
    expect(page.locator("h1")).to_contain_text("Smarter Water Quality Analysis")
    expect(page.locator("#btn-start-analysis")).to_be_visible()


def test_theme_toggle(page: Page):
    page.goto(f"{BASE_URL}/")
    theme_btn = page.locator("#theme-toggle-btn").first
    expect(theme_btn).to_be_visible()

    # Initial state
    initial_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    theme_btn.click()
    
    toggled_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert toggled_theme != initial_theme
    
    # Toggle back
    theme_btn.click()
    restored_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert restored_theme == initial_theme


def test_homepage_interactive_preview(page: Page):
    page.goto(f"{BASE_URL}/")
    
    btn_poll = page.locator("#preview-btn-poll")
    btn_poll.click()
    badge = page.locator("#prev-tier-badge")
    expect(badge).to_contain_text("Polluted")

    btn_good = page.locator("#preview-btn-good")
    btn_good.click()
    expect(badge).to_contain_text("Good")


def test_analyze_page_live_validation_and_screening(page: Page):
    page.goto(f"{BASE_URL}/analyze.html")
    expect(page.locator("h1")).to_contain_text("Water Quality Analysis Workspace")

    # Test invalid range on pH (e.g. 18.0)
    input_ph = page.locator("#input-ph")
    input_ph.fill("18.0")
    input_ph.dispatch_event("input")
    
    err_ph = page.locator("#err-ph")
    expect(err_ph).to_be_visible()
    expect(page.locator("#btn-submit-screening")).to_be_disabled()

    # Load valid template
    page.locator("#btn-load-potable").click()
    expect(err_ph).not_to_be_visible()
    expect(page.locator("#btn-submit-screening")).to_be_enabled()

    # Submit screening
    page.locator("#btn-submit-screening").click()

    # Verify assessment result container populated
    assessment = page.locator("#assessment-container")
    expect(assessment).to_be_visible(timeout=10000)
    expect(assessment).to_contain_text("Water Quality Risk")
    expect(assessment).to_contain_text("Calibrated Class Probability Distribution")

    # Verify occlusion attribution bars rendered
    attr_card = page.locator("#attribution-card")
    expect(attr_card).to_be_visible()
    expect(page.locator(".attribution-bar-fill").first).to_be_visible()


def test_batch_screening_page(page: Page):
    page.goto(f"{BASE_URL}/batch.html")
    expect(page.locator("h1")).to_contain_text("Batch Water Quality Screening")
    
    # Dropzone is present
    expect(page.locator("#batch-dropzone")).to_be_visible()

    # Load demo batch (10 samples)
    page.locator("#btn-load-sample-batch").click()

    # Summary section becomes visible
    summary = page.locator("#batch-summary-section")
    expect(summary).to_be_visible(timeout=10000)
    expect(page.locator("#batch-stat-total")).to_have_text("10")

    # Results table rendered
    table = page.locator("#batch-results-table")
    expect(table).to_be_visible()
    
    # Column sorting works
    th_tier = page.locator("th[data-col='predicted_class']")
    th_tier.click()
    expect(th_tier).to_have_class("sortable sorted-asc")


def test_results_tabs_and_lightbox(page: Page):
    page.goto(f"{BASE_URL}/results.html")
    
    # Tab navigation
    btn_benchmarks = page.locator("#btn-tab-benchmarks")
    btn_benchmarks.click()
    panel_benchmarks = page.locator("#tab-benchmarks")
    expect(panel_benchmarks).to_be_visible()

    btn_figures = page.locator("#btn-tab-figures")
    btn_figures.click()
    panel_figures = page.locator("#tab-figures")
    expect(panel_figures).to_be_visible()

    # Lightbox click on first figure
    first_fig = page.locator(".figure-container").first
    first_fig.click()
    
    modal = page.locator("#lightbox-modal")
    expect(modal).to_have_class("lightbox-modal open")
    expect(page.locator("#lightbox-img")).to_be_visible()

    # Close with Escape
    page.keyboard.press("Escape")
    expect(modal).not_to_have_class("lightbox-modal open")


def test_about_page_accordions(page: Page):
    page.goto(f"{BASE_URL}/about.html")
    expect(page.locator("h1")).to_contain_text("Research Methodology")

    # First accordion is open initially
    body_team = page.locator("#acc-body-team")
    expect(body_team).to_be_visible()
    expect(body_team).to_contain_text("Ayusman Das")

    # Second accordion is collapsed initially
    body_dataset = page.locator("#acc-body-dataset")
    expect(body_dataset).not_to_be_visible()

    # Click to expand
    header_dataset = page.locator("#acc-header-dataset")
    header_dataset.click()
    expect(body_dataset).to_be_visible()
    expect(body_dataset).to_contain_text("Kaggle Water Quality")


def test_mobile_navigation(page: Page):
    page.set_viewport_size({"width": 375, "height": 667})
    page.goto(f"{BASE_URL}/")

    toggle_btn = page.locator("#mobile-nav-btn")
    expect(toggle_btn).to_be_visible()

    nav_menu = page.locator("#nav-menu")
    expect(nav_menu).not_to_be_visible()

    toggle_btn.click()
    expect(nav_menu).to_be_visible()
