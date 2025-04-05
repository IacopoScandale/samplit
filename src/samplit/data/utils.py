import time
from contextlib import contextmanager


# only to time pipeline functions within terminal
@contextmanager
def time_it(description: str):
  print(f"\n{description}")
  start_time = time.time()  # Start timer
  yield  # Qui viene eseguito il codice all'interno del blocco `with`
  end_time = time.time()  # End timer
  execution_time = end_time - start_time
  if execution_time < 60:
    print(f"\n  done! ({execution_time:.4f} sec)\n")
  else:
    print(f"\n  done! ({execution_time/60:.1f} min)\n")
