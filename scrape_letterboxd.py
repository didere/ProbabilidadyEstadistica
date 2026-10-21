"""
Scraper de reseñas de "Niu Lai" en Letterboxd + muestreo
  - Muestreo por conveniencia: las primeras CONV_PAGES páginas (lo más fácil de alcanzar)
  - Muestreo aleatorio simple: n reseñas elegidas al azar de TODAS las páginas descargadas
"""
import time
import pandas as pd
from playwright.sync_api import sync_playwright

FILM_SLUG = "niu-lai"
MAX_PAGES = 15            # páginas a descargar para formar la "población"
CONV_PAGES = 2            # páginas que forman la muestra por conveniencia
N_RANDOM = 100            # tamaño de la muestra aleatoria
SEED = 42                 # semilla para que el muestreo sea reproducible

EXTRACT_JS = """    
() => {
  const items = document.querySelectorAll(
    'li.film-detail, li.production-viewing, article.production-viewing, div.film-detail'
  );
  const out = [];
  items.forEach(el => {
    const authorEl = el.querySelector('.displayname, .name');
    const author = authorEl ? authorEl.textContent.trim() : null;

    // Clase "rated-N": N va de 1 a 10 (medias estrellas) -> estrellas = N / 2
    let rating = null;

    // Estrategia 0 (diseño actual): <svg class="glyph -rating" aria-label="★★★½">
    const glyph = el.querySelector('svg.-rating, .glyph.-rating, [aria-label*="★"]');
    if (glyph) {
      const label = glyph.getAttribute('aria-label') ||
                    (glyph.querySelector('title') ? glyph.querySelector('title').textContent : '');
      const stars = (label.match(/★/g) || []).length;
      if (stars > 0 || label.includes('½')) rating = stars + (label.includes('½') ? 0.5 : 0);
    }

    // Estrategia 1: cualquier elemento con clase rated-N (o rated-large-N) dentro de la reseña
    const rated = rating === null ? el.querySelector('[class*="rated-"]') : null;
    if (rated) {
      const m = (rated.getAttribute('class') || '').match(/rated-(?:large-)?(\\d+)/);
      if (m) rating = parseInt(m[1], 10) / 2;
    }

    // Estrategia 2: atributos tipo data-rating / aria-label ("4.5 stars")
    if (rating === null) {
      const attrEl = el.querySelector('[data-rating], [aria-label*="star"], [title*="star"]');
      if (attrEl) {
        const raw = attrEl.getAttribute('data-rating') ||
                    attrEl.getAttribute('aria-label') || attrEl.getAttribute('title');
        const m = raw && raw.match(/(\\d+(?:\\.\\d+)?)/);
        if (m) {
          const v = parseFloat(m[1]);
          rating = v > 5 ? v / 2 : v;
        }
      }
    }

    // Estrategia 3: contar los símbolos ★ y ½ del texto
    if (rating === null) {
      const starEl = el.querySelector('.rating, .content-metadata');
      const t = starEl ? starEl.textContent : '';
      const stars = (t.match(/★/g) || []).length;
      if (stars > 0 || t.includes('½')) rating = stars + (t.includes('½') ? 0.5 : 0);
    }

    // textContent (no innerText) para no perder texto colapsado tras "more"
    const bodyEl = el.querySelector('.body-text, .js-review-body');
    const text = bodyEl ? bodyEl.textContent.trim() : '';

    const dateEl = el.querySelector('time, .date, .timestamp');
    let date = null;
    if (dateEl) date = dateEl.getAttribute('datetime') || dateEl.textContent.trim();

    if (text || rating !== null) {
      out.push({author, rating_out_of_5: rating, review_text: text, review_date: date});
    }
  });
  return out;
}
"""


def scrape_reviews(max_pages=MAX_PAGES):
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        for n in range(1, max_pages + 1):
            url = f"https://letterboxd.com/film/{FILM_SLUG}/reviews/page/{n}/"
            print(f"Cargando página {n}: {url}")
            try:
                resp = page.goto(url, timeout=30000, wait_until="domcontentloaded")
                if resp is not None and resp.status == 404:
                    print("404: el slug de la película no existe. Revisa FILM_SLUG.")
                    break
                try:
                    page.wait_for_selector(
                        "li.film-detail, li.production-viewing, article.production-viewing",
                        timeout=15000,
                    )
                except Exception:
                    pass

                data = page.evaluate(EXTRACT_JS)

                if not data:
                    print(f"Sin reseñas en página {n}. Título de la página: {page.title()!r}")
                    page.screenshot(path=f"debug_page_{n}.png")
                    with open(f"debug_page_{n}.html", "w", encoding="utf-8") as f:
                        f.write(page.content())
                    print(f"Guardé debug_page_{n}.png/.html para inspeccionar.")
                    break

                for d in data:
                    d.update(movie_title="Niu Lai", platform="Letterboxd", page=n)
                rows.extend(data)
                time.sleep(2)  # pausa cortés entre páginas

            except Exception as err:
                print(f"Error en página {n}: {err}")
                break

        browser.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["author", "review_text", "review_date"]).reset_index(drop=True)
    return df


def build_samples(df):
    conv = df[df["page"] <= CONV_PAGES].copy()
    rand = df.sample(n=min(N_RANDOM, len(df)), random_state=SEED).copy()
    return conv, rand


if __name__ == "__main__":
    population = scrape_reviews()

    if population.empty:
        print("\nNo se pudieron extraer reseñas. Revisa los archivos debug_page_*.")
    else:
        conv, rand = build_samples(population)
        population.to_csv("niu_lai_poblacion.csv", index=False, encoding="utf-8-sig")
        conv.to_csv("niu_lai_muestra_conveniencia.csv", index=False, encoding="utf-8-sig")
        rand.to_csv("niu_lai_muestra_aleatoria.csv", index=False, encoding="utf-8-sig")

        print(f"\nPoblación descargada: {len(population)} reseñas")
        print(f"Reseñas con rating: {population['rating_out_of_5'].notna().sum()}")
        print(f"Muestra por conveniencia: {len(conv)} | Muestra aleatoria: {len(rand)}")
        print("\nRating promedio (estrellas sobre 5):")
        print(f"  Población:     {population['rating_out_of_5'].mean():.2f}")
        print(f"  Conveniencia:  {conv['rating_out_of_5'].mean():.2f}")
        print(f"  Aleatoria:     {rand['rating_out_of_5'].mean():.2f}")
        print(population.head())