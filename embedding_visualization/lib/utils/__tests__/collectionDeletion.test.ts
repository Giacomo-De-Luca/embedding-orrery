import { describe, it, expect, vi } from 'vitest';
import { runDeleteCollection } from '../collectionDeletion';

describe('runDeleteCollection', () => {
  it('resolves to whether the backend deleted the collection', async () => {
    const deleted = vi.fn().mockResolvedValue({ data: { deleteCollection: true } });
    const missing = vi.fn().mockResolvedValue({ data: { deleteCollection: false } });

    await expect(runDeleteCollection(deleted, 'poems')).resolves.toBe(true);
    await expect(runDeleteCollection(missing, 'poems')).resolves.toBe(false);
    expect(deleted).toHaveBeenCalledWith({ variables: { collectionName: 'poems' } });
  });

  it("rejects with the server's message, e.g. a running job to cancel first", async () => {
    const message = "Collection 'poems' has a running job (poems_probe); cancel it before deleting the collection";
    const returned = vi.fn().mockResolvedValue({ data: null, error: new Error(message) });
    const thrown = vi.fn().mockRejectedValue(new Error(message));

    await expect(runDeleteCollection(returned, 'poems')).rejects.toThrow(message);
    await expect(runDeleteCollection(thrown, 'poems')).rejects.toThrow(message);
  });

  it('treats a missing result as not deleted', async () => {
    const empty = vi.fn().mockResolvedValue({ data: null });
    await expect(runDeleteCollection(empty, 'poems')).resolves.toBe(false);
  });
});
