<?php

namespace TeamTNT\TNTSearch\Classifier;

use TeamTNT\TNTSearch\Stemmer\NoStemmer;
use TeamTNT\TNTSearch\Support\Tokenizer;

class TNTClassifier
{
    public $documents              = [];
    public $words                  = [];
    public $types                  = [];
    public $tokenizer              = null;
    public $stemmer                = null;
    protected $arraySumOfWordType  = null;
    protected $arraySumOfDocuments = null;

    public function __construct()
    {
        $this->tokenizer = new Tokenizer;
        $this->stemmer   = new NoStemmer;
    }

    //From https://gist.github.com/raymondjplante/d826df05349c1d4350e0aa2d7ca01da4
    function softmax(array $v){

        //Just in case values are passed in as string, apply floatval
        $v = array_map('exp',array_map('floatval',$v));
        $sum = array_sum($v);

        foreach($v as $index => $value) {
            $v[$index] = $value/$sum;
        }

        return $v;
    }

    //From https://stackoverflow.com/a/54867175/9407664
    public function predict($statement)
    {
        $words = $this->tokenizer->tokenize($statement);

        $best_likelihoods = [];
        $best_likelihood = -INF;
        $best_type       = '';
        foreach ($this->types as $type) {
            $best_likelihoods[$type] = -INF;
            $likelihood = log($this->pTotal($type)); // calculate P(Type)
            $p          = 0;
            foreach ($words as $word) {
                $word = $this->stemmer->stem($word);
                $p += log($this->p($word, $type));
            }
            $likelihood += $p; // calculate P(word, Type)
            if ($likelihood > $best_likelihood) {
                $best_likelihood = $likelihood;
                $best_likelihoods[$type] = $likelihood;
                $best_type       = $type;
            }
        }

        return [
            'likelihood' => $best_likelihood,
            'likelihoods' => $best_likelihoods,
            'probability' => $this->softmax($best_likelihoods),
            'label'      => $best_type
        ];
    }

    public function learn($statement, $type)
    {
        if (!in_array($type, $this->types)) {
            $this->types[] = $type;
        }

        $words = $this->tokenizer->tokenize($statement);

        foreach ($words as $word) {
            $word = $this->stemmer->stem($word);

            if (!isset($this->words[$type][$word])) {
                $this->words[$type][$word] = 0;
            }
            $this->words[$type][$word]++; // increment the word count for the type
        }
        if (!isset($this->documents[$type])) {
            $this->documents[$type] = 0;
        }

        $this->documents[$type]++; // increment the document count for the type
    }

    public function p($word, $type)
    {
        $count = 0;
        if (isset($this->words[$type][$word])) {
            $count = $this->words[$type][$word];
        }

        if (!isset($this->arraySumOfWordType[$type])) {
            $this->arraySumOfWordType[$type] = array_sum($this->words[$type]);
        }

        return ($count + 1) / ($this->arraySumOfWordType[$type] + $this->vocabularyCount());
    }

    public function pTotal($type)
    {
        if (!isset($this->arraySumOfDocuments)) {
            $this->arraySumOfDocuments = array_sum($this->documents);
        }
        return ($this->documents[$type]) / $this->arraySumOfDocuments;
    }

    public function vocabularyCount()
    {
        if (isset($this->vc)) {
            return $this->vc;
        }

        $words = [];
        foreach ($this->words as $key => $value) {
            foreach ($this->words[$key] as $word => $count) {
                $words[$word] = 0;
            }
        }
        $this->vc = count($words);
        return $this->vc;
    }

    public function save($path)
    {
        $s = serialize($this);
        return file_put_contents($path, $s);
    }

    public function load($name)
    {
        $s          = file_get_contents($name);
        $classifier = unserialize($s);

        unset($this->vc);
        unset($this->arraySumOfDocuments);
        unset($this->arraySumOfWordType);

        $this->documents = $classifier->documents;
        $this->words     = $classifier->words;
        $this->types     = $classifier->types;
        $this->tokenizer = $classifier->tokenizer;
        $this->stemmer   = $classifier->stemmer;
    }
}
