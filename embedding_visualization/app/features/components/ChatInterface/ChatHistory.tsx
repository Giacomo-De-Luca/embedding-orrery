'use client';

import { memo, useState } from 'react';
import { MoreHorizontal, Plus, Sparkles, Trash2 } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/lib/ui-primitives/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import type { ChatSessionSummary } from '@/lib/types/types';
import { groupSessionsByDate, type GroupedSessions } from '../../utils/chatHistoryUtils';
import { SteeringIdenticon } from './SteeringIdenticon';

interface ChatHistoryProps {
  sessions: ChatSessionSummary[];
  loading: boolean;
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onNewChat: () => void;
}

const ChatHistoryItem = memo(function ChatHistoryItem({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session: ChatSessionSummary;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={`group flex items-center gap-1.5 rounded-md px-2 py-1 cursor-pointer transition-colors ${
        isActive
          ? 'bg-accent text-accent-foreground'
          : 'hover:bg-muted/50'
      }`}
      onClick={onSelect}
    >
      <SteeringIdenticon
        features={session.config.features}
        size={14}
        fallback={<Sparkles className="size-3 text-muted-foreground" />}
        className="shrink-0"
        crossfadeOnChange
      />
      <span className="flex-1 truncate text-xs">{session.title}</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
          <Button
            variant="ghost"
            size="icon"
            className="h-5 w-5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <MoreHorizontal className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onClick={onDelete}
          >
            <Trash2 className="mr-2 h-3.5 w-3.5" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
});

function SessionGroup({
  label,
  sessions,
  activeSessionId,
  onSelectSession,
  onRequestDelete,
}: {
  label: string;
  sessions: ChatSessionSummary[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onRequestDelete: (id: string) => void;
}) {
  if (sessions.length === 0) return null;
  return (
    <div className="mb-2">
      <div className="px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      {sessions.map((session) => (
        <ChatHistoryItem
          key={session.id}
          session={session}
          isActive={session.id === activeSessionId}
          onSelect={() => onSelectSession(session.id)}
          onDelete={() => onRequestDelete(session.id)}
        />
      ))}
    </div>
  );
}

export function ChatHistory({
  sessions,
  loading,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  onNewChat,
}: ChatHistoryProps) {
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const grouped: GroupedSessions = groupSessionsByDate(sessions);

  const handleConfirmDelete = () => {
    if (deleteId) {
      onDeleteSession(deleteId);
      setDeleteId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 overflow-y-auto px-2 py-3">
        <div className="flex flex-col gap-0.5">
          {[44, 32, 28, 64, 52].map((w, i) => (
            <div key={i} className="flex h-6 items-center rounded-md px-2">
              <div
                className="h-2.5 animate-pulse rounded-sm bg-muted"
                style={{ width: `${w}%` }}
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        <Button
          variant="outline"
          size="sm"
          className="mb-2 w-full justify-start gap-1.5 text-xs h-7"
          onClick={onNewChat}
        >
          <Plus className="h-3 w-3" />
          New Chat
        </Button>

        {sessions.length === 0 ? (
          <div className="mt-8 text-center text-sm text-muted-foreground">
            Your conversations will appear here once you start chatting.
          </div>
        ) : (
          <>
            <SessionGroup
              label="Today"
              sessions={grouped.today}
              activeSessionId={activeSessionId}
              onSelectSession={onSelectSession}
              onRequestDelete={setDeleteId}
            />
            <SessionGroup
              label="Yesterday"
              sessions={grouped.yesterday}
              activeSessionId={activeSessionId}
              onSelectSession={onSelectSession}
              onRequestDelete={setDeleteId}
            />
            <SessionGroup
              label="Last 7 days"
              sessions={grouped.lastWeek}
              activeSessionId={activeSessionId}
              onSelectSession={onSelectSession}
              onRequestDelete={setDeleteId}
            />
            <SessionGroup
              label="Last 30 days"
              sessions={grouped.lastMonth}
              activeSessionId={activeSessionId}
              onSelectSession={onSelectSession}
              onRequestDelete={setDeleteId}
            />
            <SessionGroup
              label="Older"
              sessions={grouped.older}
              activeSessionId={activeSessionId}
              onSelectSession={onSelectSession}
              onRequestDelete={setDeleteId}
            />
          </>
        )}
      </div>

      <Dialog open={deleteId !== null} onOpenChange={(open) => !open && setDeleteId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete conversation?</DialogTitle>
            <DialogDescription>
              This will permanently delete this conversation and all its messages.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleConfirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
