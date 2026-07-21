/**
 * Apollo Client configuration for GraphQL backend
 *
 * Supports:
 * - HTTP queries and mutations
 * - WebSocket subscriptions for real-time progress updates
 */

import {
  ApolloClient,
  InMemoryCache,
  HttpLink,
  split,
} from '@apollo/client';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { getMainDefinition } from '@apollo/client/utilities';
import { createClient } from 'graphql-ws';
import {
  resolveGraphqlHttpUrl,
  resolveGraphqlWebSocketUrl,
} from './endpointUrls';

// HTTP endpoint for queries and mutations
const httpLink = new HttpLink({
  uri: resolveGraphqlHttpUrl(process.env.NEXT_PUBLIC_GRAPHQL_URL),
});

// WebSocket endpoint for subscriptions
// Only create WebSocket link on client side
const wsLink =
  typeof window !== 'undefined'
    ? new GraphQLWsLink(
        createClient({
          url: resolveGraphqlWebSocketUrl(
            process.env.NEXT_PUBLIC_GRAPHQL_WS_URL,
            window.location,
          ),
          // Reconnect on connection loss
          retryAttempts: 5,
          shouldRetry: () => true,
        })
      )
    : null;

// Split link: use WebSocket for subscriptions, HTTP for everything else
const splitLink =
  typeof window !== 'undefined' && wsLink
    ? split(
        ({ query }) => {
          const definition = getMainDefinition(query);
          return (
            definition.kind === 'OperationDefinition' &&
            definition.operation === 'subscription'
          );
        },
        wsLink,
        httpLink
      )
    : httpLink;

export const apolloClient = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
    },
    query: {
      fetchPolicy: 'cache-first',
    },
  },
});
