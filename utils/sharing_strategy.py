from heapq import nlargest
from heapq import nsmallest

def min_loss(clients_loss, n_smallest):
	smallest_loss_clients = nsmallest(N, clients_loss, key=clients_loss.get)
	return smallest_loss_clients

def best_test_acc(clients_acc, n_largest):
	largest_acc_clients = nlargest(N, clients_acc, key=clients_acc.get)
	return largest_acc_clients
