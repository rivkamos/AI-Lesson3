export interface Payment {
  id: string;
  tenantId: string;
  amount: number;
  dueDate: Date;
  status: 'paid' | 'pending' | 'late';
}