import { isToday, isYesterday, subMonths, subWeeks } from 'date-fns';
import type { ChatSessionSummary } from '@/lib/types/types';

export type GroupedSessions = {
  today: ChatSessionSummary[];
  yesterday: ChatSessionSummary[];
  lastWeek: ChatSessionSummary[];
  lastMonth: ChatSessionSummary[];
  older: ChatSessionSummary[];
};

export function groupSessionsByDate(sessions: ChatSessionSummary[]): GroupedSessions {
  const now = new Date();
  const oneWeekAgo = subWeeks(now, 1);
  const oneMonthAgo = subMonths(now, 1);

  return sessions.reduce(
    (groups, session) => {
      const date = new Date(session.updatedAt);

      if (isToday(date)) {
        groups.today.push(session);
      } else if (isYesterday(date)) {
        groups.yesterday.push(session);
      } else if (date > oneWeekAgo) {
        groups.lastWeek.push(session);
      } else if (date > oneMonthAgo) {
        groups.lastMonth.push(session);
      } else {
        groups.older.push(session);
      }

      return groups;
    },
    {
      today: [],
      yesterday: [],
      lastWeek: [],
      lastMonth: [],
      older: [],
    } as GroupedSessions
  );
}
