echo '100000 300' > wiki.es.100k.align.vec
(head -n 100001 wiki.es.align.vec | tail -n 100000) >> wiki.es.100k.align.vec
echo '100000 300' > wiki.en.100k.align.vec
(head -n 100001 wiki.en.align.vec | tail -n 100000) >> wiki.en.100k.align.vec
echo '100000 300' > wiki.it.100k.align.vec
(head -n 100001 wiki.it.align.vec | tail -n 100000) >> wiki.it.100k.align.vec