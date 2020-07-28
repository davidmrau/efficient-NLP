from transformers import pipeline



from transformers import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')



#model_folder = 'experiments/lm_batch_8'
model_folder = 'experiments/lm_batch_8/checkpoint-56000/'
fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer_bert
    #tokenizer=model_folder
)

# The sun <mask>.
# =>
#result = fill_mask("La suno [MASK].")
result = fill_mask('which chromosome controls sex characteristics sex chromosome [SEP] - (genetics) a chromosome that determines the sex of an individual; mammals normally have two sex chromosomes chromosome - a threadlike strand of [MASK] in the cell nucleus that carries the genes in a linear order; humans have 22 chromosome pairs plus two sex chromosomes.')

result = fill_mask('what is the name of the cat from that show [SEP] the [MASK]\'s is John [SEP]')
print(result)
#print(result)
