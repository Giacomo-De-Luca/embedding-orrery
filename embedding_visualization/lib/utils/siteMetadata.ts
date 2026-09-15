/**
 * Site-wide metadata: the document title/description and the public origin
 * used to make social-preview URLs absolute.
 *
 * Open Graph / Twitter crawlers require an absolute `og:image` URL. Next.js
 * derives it from `metadata.metadataBase`; when that is unset, production
 * builds fall back to `http://localhost:3000`, which no crawler can fetch.
 * The origin is a build-time constant (`NEXT_PUBLIC_SITE_URL`, a Docker build
 * arg). For the HF Space, `deploy/hf-space/deploy.py` stores it as a Space
 * Variable, which Hugging Face passes to the Docker build as a build-arg.
 * See documentation/HF_SPACE_DEMO.md ("Social preview").
 */

export const SITE_NAME = 'Orrery';
export const SITE_TITLE = 'Orrery: Embedding Observatory';
export const SITE_DESCRIPTION =
  'Explore embedding spaces as interactive 3D constellations: topic clusters, semantic search, and sparse-autoencoder features.';

const SCHEME_PATTERN = /^[a-z][a-z0-9+.-]*:\/\//i;

/**
 * Parse a configured public origin into a URL usable as `metadataBase`.
 *
 * - empty / whitespace / unset → `undefined` (Next falls back to localhost)
 * - a bare host (`my-space.hf.space`) → `https://my-space.hf.space/`
 * - anything that is not an http(s) URL → `undefined`
 */
export function resolveSiteUrl(raw: string | undefined | null): URL | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;
  const candidate = SCHEME_PATTERN.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    const url = new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return undefined;
    return url;
  } catch {
    return undefined;
  }
}
