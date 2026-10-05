import { Routes } from '@angular/router';
import { Dashboard } from '../components/dashboard/dashboard';
import { Shell } from '../components/layout/shell/shell';
import { AiChat } from '../components/ai-chat/ai-chat';

export const routes: Routes = [{
    path: '',
    component: Shell,
    children: [
        {
            path: '',
            component: Dashboard
        },{
    path: 'ai-chat',
    component: AiChat
},
{path: 'buildings', 
    loadComponent: () => import('../components/buildings/buildings')
    .then(m => m.Buildings)},
    {path: 'tenants', 
    loadComponent: () => import('../components/tenants/tenants')
    .then(m => m.Tenants)},
    {path: 'payments', 
    loadComponent: () => import('../components/payments/payments')
    .then(m => m.Payments)} 
    ]}, 


];
