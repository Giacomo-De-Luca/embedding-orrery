/**
 * The deleteCollection mutation, read for the delete dialog.
 *
 * Resolves to whether the backend deleted the collection (false: it existed
 * nowhere, or a step failed). Rejects with the server's message when the
 * mutation errors, e.g. while one of the collection's jobs is running, so the
 * dialog can tell the user to cancel it first instead of a generic failure.
 */
export async function runDeleteCollection(
  mutate: (options: { variables: { collectionName: string } }) => Promise<{
    data?: { deleteCollection?: boolean } | null;
    error?: { message: string } | null;
  }>,
  collectionName: string,
): Promise<boolean> {
  const { data, error } = await mutate({ variables: { collectionName } });
  if (error) throw new Error(error.message);
  return data?.deleteCollection ?? false;
}
