The contents of this directory are supposed to demonstrate how to use the parallelism machinery. In more detail, we imagine a simple calculation that can be implemented with the parallelism machinery. We start with a simple implementation and we make incremental changes until we arrive at something that works with our parallelism machinery.

We actually implement 2 versions of this logic. One version where the problem has an implicit chunking that we can take advantage for better performance. The other version, has no ordering.
