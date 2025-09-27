type ParseInt<T extends string> = T extends `${infer Digit extends number}`
  ? Digit
  : never;

type ReverseString<S extends string> = S extends `${infer Frist}${infer Rest}`
  ? `${ReverseString<Rest>}${Frist}`
  : "";

type RemoveLeadingZeros<S extends string> = S extends "0"
  ? S
  : S extends `0${infer R}`
  ? RemoveLeadingZeros<R>
  : S;

type PutSign<S extends string> = `-${S}`;

// `S` is a reversed string representing some number, e.g., "0321" instead of 1230
type InternalMinusOne<S extends string> =
  S extends `${infer Digit extends number}${infer Rest}`
    ? Digit extends 0
      ? `9${InternalMinusOne<Rest>}`
      : `${[9, 0, 1, 2, 3, 4, 5, 6, 7, 8][Digit]}${Rest}`
    : never;

type AlmostFullMinusOne<T extends number> = ParseInt<
  RemoveLeadingZeros<ReverseString<InternalMinusOne<ReverseString<`${T}`>>>>
>;

type InternalPlusOne<S extends string> = S extends "9"
  ? "01"
  : S extends `${infer Digit extends number}${infer Rest}`
  ? Digit extends 9
    ? `0${InternalPlusOne<Rest>}`
    : `${[1, 2, 3, 4, 5, 6, 7, 8, 9][Digit]}${Rest}`
  : never;

type MinusOne<T extends number> = T extends 0
  ? -1 // T = 0
  : `${T}` extends `-${infer Abs}`
  ? ParseInt<PutSign<ReverseString<InternalPlusOne<ReverseString<Abs>>>>> // T < 0
  : // T > 0
    ParseInt<
      RemoveLeadingZeros<ReverseString<InternalMinusOne<ReverseString<`${T}`>>>>
    >;

// TODO: fix this one
type PlusOne<T extends number> = T extends PutSign<"1">
  ? 0 // T = -1
  : `${T}` extends `-${infer Abs}`
  ? // T < -1
    ParseInt<
      PutSign<
        RemoveLeadingZeros<ReverseString<InternalMinusOne<ReverseString<Abs>>>>
      >
    >
  : // T >= 0
    ParseInt<
      RemoveLeadingZeros<ReverseString<InternalPlusOne<ReverseString<`${T}`>>>>
    >;

type nn = MinusOne<100>;
// type MinusOne<n extends number>  = n extends 0?
type Repeat<
  V extends string,
  count extends number,
  acc extends string = ""
> = count extends 0 ? acc : Repeat<V, MinusOne<count>, `${acc}${V}`>;

type Create10ToPower<Power extends number> = ParseInt<`1${Repeat<"0", Power>}`>;

type Take<S extends string, n extends number> = n extends 0
  ? ""
  : S extends `${infer First}${infer Rest}`
  ? `${First}${Take<Rest, MinusOne<n>>}`
  : S;

type t = Take<"1234", 5>;

type LenString<S extends string> = S extends ""
  ? 0
  : S extends `${infer First}${infer Rest}`
  ? PlusOne<LenString<Rest>>
  : 0;

type AllZeros<P extends string> = P extends Repeat<"0", LenString<P>>
  ? true
  : false;

type ReplacePre<S extends string, R extends string> = S extends `${Take<
  S,
  LenString<R>
>}${infer Rest}`
  ? `${R}${Rest}`
  : S;

type xxx = ReplacePre<"0000001", "321">;

type AddToPowerOfTen<P extends string, N extends string> = ParseInt<
  ReverseString<ReplacePre<ReverseString<P>, ReverseString<N>>>
>;

type addf = AddToPowerOfTen<"10000", "123">;
