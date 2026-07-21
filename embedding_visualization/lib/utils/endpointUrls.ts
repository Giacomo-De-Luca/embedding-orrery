export interface BrowserLocation {
  protocol: string;
  host: string;
}

const LOCAL_GRAPHQL_HTTP_URL = 'http://localhost:8000/graphql';
const LOCAL_GRAPHQL_WS_URL = 'ws://localhost:8000/graphql';
const LOCAL_API_BASE_URL = 'http://localhost:8000';

function configuredValue(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

export function resolveGraphqlHttpUrl(configured: string | null | undefined): string {
  if (configured === undefined || configured === null) {
    return LOCAL_GRAPHQL_HTTP_URL;
  }
  return configuredValue(configured) ?? '/graphql';
}

export function resolveGraphqlWebSocketUrl(
  configured: string | null | undefined,
  location: BrowserLocation,
): string {
  if (configured === undefined || configured === null) {
    return LOCAL_GRAPHQL_WS_URL;
  }
  const explicit = configuredValue(configured);
  if (explicit) return explicit;
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/graphql`;
}

export function resolveApiBaseUrl(configured: string | null | undefined): string {
  if (configured === undefined || configured === null) {
    return LOCAL_API_BASE_URL;
  }
  return (configuredValue(configured) ?? '').replace(/\/$/, '');
}
